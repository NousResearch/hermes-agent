"""Execute the sidecar's poll identity and tally logic under Node."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


_MODULE = Path("plugins/platforms/photon/sidecar/poll-votes.mjs").resolve().as_uri()


def _node(script: str):
    result = subprocess.run(
        ["node", "--input-type=module", "-e",
         f"import {{PollVoteTracker,parsePollVoteId}} from {json.dumps(_MODULE)};\n" + script],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_poll_id_parsing_and_direct_fields() -> None:
    result = _node("""
      const tracker = new PollVoteTracker();
      const message = {id:'poll-guid:mailto:user@example.com:option:selected:123',
        sender:{id:'mailto:user@example.com'}};
      const content = {type:'poll_option',option:{title:'A'},selected:true};
      const direct = [
        {pollId:'direct'}, {pollMessageGuid:'guid'}, {poll:{id:'nested'}},
        {poll:{pollMessageGuid:'nested-guid'}}, {pollId:'',poll:{messageId:'message'}}
      ].map(fields => tracker.normalize({...content,...fields},message).pollId);
      console.log(JSON.stringify({
        parsed:parsePollVoteId(message.id,message.sender.id),
        noSender:parsePollVoteId(message.id), direct,
        inferred:tracker.normalize(content,message).pollId,
        messageField:tracker.normalize(content,{...message,pollId:'message-direct'}).pollId,
        malformed:['plain-id','poll:voter:option:unexpected:123',
          'poll:voter:option:selected:NaN',':voter:option:selected:123',
          'option:selected:123','poll:option:selected:123'].map(id=>parsePollVoteId(id))
      }));
    """)
    assert result == {
        "parsed": {"pollId": "poll-guid", "eventTime": 123},
        "noSender": {"pollId": "poll-guid", "eventTime": 123},
        "direct": ["direct", "guid", "nested", "nested-guid", "message"],
        "inferred": "poll-guid",
        "messageField": "message-direct",
        "malformed": [None] * 6,
    }


def test_tally_tracks_multiple_voters_and_deselect_reselect() -> None:
    result = _node("""
      const tracker = new PollVoteTracker();
      let time=0;
      const vote=(voter,title,selected=true,pollId='poll')=>tracker.normalize({
        type:'poll_option',option:{title},selected,
        poll:{title:'Which?',options:[{title:'A'},{title:'B'}]}
      },{id:`${pollId}:${voter}:${title}:${selected?'selected':'deselected'}:${++time}`,
        sender:{id:voter}});
      const snapshot=event=>({pollId:event.pollId,tally:event.tally,voters:event.voters});
      console.log(JSON.stringify([
        snapshot(vote('alice','A')), snapshot(vote('alice','A')),
        snapshot(vote('bob','A')), snapshot(vote('alice','B')),
        snapshot(vote('alice','A',false)), snapshot(vote('alice','B',false)),
        snapshot(vote('alice','A')), snapshot(vote('carol','B',false)),
        snapshot(vote('alice','B',true,'other'))
      ]));
    """)
    assert result == [
        {"pollId": "poll", "tally": {"A": 1, "B": 0}, "voters": 1},
        {"pollId": "poll", "tally": {"A": 1, "B": 0}, "voters": 1},
        {"pollId": "poll", "tally": {"A": 2, "B": 0}, "voters": 2},
        {"pollId": "poll", "tally": {"A": 2, "B": 1}, "voters": 2},
        {"pollId": "poll", "tally": {"A": 1, "B": 1}, "voters": 2},
        {"pollId": "poll", "tally": {"A": 1, "B": 0}, "voters": 1},
        {"pollId": "poll", "tally": {"A": 2, "B": 0}, "voters": 2},
        {"pollId": "poll", "tally": {"A": 2, "B": 0}, "voters": 2},
        {"pollId": "other", "tally": {"A": 0, "B": 1}, "voters": 1},
    ]


def test_replay_does_not_restore_an_older_selection() -> None:
    result = _node("""
      const tracker=new PollVoteTracker();
      const vote=(selected,time)=>tracker.normalize({
        type:'poll_option',option:{title:'A'},selected
      },{id:`poll:voter:option:${selected?'selected':'deselected'}:${time}`,
        sender:{id:'voter'}});
      vote(true,1); vote(false,2);
      const replay=vote(true,1);
      const changed=vote(true,3);
      console.log(JSON.stringify({replay,changed}));
    """)
    assert result["replay"]["tally"] == {"A": 0}
    assert result["replay"]["voters"] == 0
    assert result["changed"]["tally"] == {"A": 1}
    assert result["changed"]["voters"] == 1


def test_tallies_evict_least_recent_poll_and_reset_on_restart() -> None:
    result = _node("""
      const tracker=new PollVoteTracker(2);
      const vote=(tracker,pollId,voter)=>tracker.normalize({
        type:'poll_option',pollId,option:{title:'A'},selected:true
      },{sender:{id:voter},timestamp:'2026-10-05T12:00:00Z'});
      vote(tracker,'one','alice'); vote(tracker,'two','alice');
      vote(tracker,'one','bob'); vote(tracker,'three','alice');
      const recent=vote(tracker,'one','carol');
      const evicted=vote(tracker,'two','bob');
      const restarted=vote(new PollVoteTracker(),'one','carol');
      console.log(JSON.stringify({recent,evicted,restarted}));
    """)
    assert result["recent"]["tally"] == {"A": 3}
    assert result["evicted"]["tally"] == {"A": 1}
    assert result["restarted"]["tally"] == {"A": 1}


def test_missing_identity_and_special_option_titles() -> None:
    result = _node("""
      const tracker=new PollVoteTracker();
      const noPoll=tracker.normalize({option:{title:'A'}},{sender:{id:'alice'}});
      const noVoter=tracker.normalize({pollId:'poll',option:{title:'A'}});
      const special=tracker.normalize({pollId:'poll',option:{title:'__proto__'}},
        {sender:{id:'alice'}});
      console.log(JSON.stringify({noPoll,noVoter,special}));
    """)
    assert result["noPoll"]["pollId"] is None
    assert result["noPoll"]["tally"] == {}
    assert result["noVoter"]["tally"] == {"A": 0}
    assert result["noVoter"]["voters"] == 0
    assert result["special"]["tally"] == {"A": 0, "__proto__": 1}


def test_same_millisecond_replay_keeps_the_later_deselection() -> None:
    """Ported from the adversarial run: select seq21, unselect seq22 at the same millisecond,
    then a reconnect replays seq21. The replay must not restore the removed vote."""
    result = _node("""
      const tracker=new PollVoteTracker();
      const vote=(voter,title,selected)=>tracker.normalize({
        type:'poll_option',option:{title},selected,
        poll:{title:'Poll',options:[{title:'Route'},{title:'Calendar'}]}
      },{id:`poll:${voter}:${title}:${selected?'selected':'deselected'}:1000`,sender:{id:voter}});
      vote('alice','Route',true); vote('alice','Route',false);
      const replay=vote('alice','Route',true);
      const other=vote('bob','Calendar',true);
      console.log(JSON.stringify({replay,other}));
    """)
    assert result["replay"]["tally"] == {"Route": 0, "Calendar": 0}
    assert result["other"]["tally"] == {"Route": 0, "Calendar": 1}
    assert result["other"]["voters"] == 1


def test_totals_are_partial_unless_poll_seen_from_creation() -> None:
    result = _node("""
      const tracker=new PollVoteTracker();
      tracker.noteCreated('sent'); tracker.noteCreated(null);
      const vote=pollId=>tracker.normalize({type:'poll_option',pollId,option:{title:'A'}},
        {sender:{id:'alice'},timestamp:'2026-10-05T12:00:00Z'});
      console.log(JSON.stringify({sent:vote('sent').partial,old:vote('old').partial,
        none:tracker.normalize({option:{title:'A'}}).partial}));
    """)
    assert result == {"sent": False, "old": True, "none": True}


_SEEDS = Path("plugins/platforms/photon/sidecar/poll-seeds.mjs").resolve().as_uri()


def _seed_node(script: str, cwd: Path):
    result = subprocess.run(
        ["node", "--input-type=module", "-e",
         f"import {{PollSeedStore}} from {json.dumps(_SEEDS)};\n" + script],
        capture_output=True, text=True, check=False, cwd=cwd,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), result.stderr


def test_poll_seeds_survive_restart_bounded_and_atomic(tmp_path: Path) -> None:
    state = tmp_path / "state" / "poll-seeds.json"
    first, _ = _seed_node(f"""
      const store=new PollSeedStore({json.dumps(str(state))},2);
      store.remember('one','Which?',[{{text:'Route',optionIdentifier:'a'}},{{text:'Calendar',optionIdentifier:'b'}}]);
      store.remember('two','',[{{text:'Route',optionIdentifier:'c'}},{{text:'Calendar'}}]);
      store.remember('three','Q',[{{text:'X',optionIdentifier:'x'}}]);
      store.remember('none','Q',[{{text:'X'}}]);
      console.log(JSON.stringify(null));
    """, tmp_path)
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert [poll["id"] for poll in saved["polls"]] == ["two", "three"]
    assert [path.name for path in state.parent.iterdir()] == ["poll-seeds.json"]  # no temp left

    reloaded, _ = _seed_node(f"""
      const store=new PollSeedStore({json.dumps(str(state))}).load();
      const two=store.get('two');
      console.log(JSON.stringify({{title:two.poll.title,ids:[...two.optionsByIdentifier]
        .map(([id,option])=>[id,option.title]),one:store.get('one')??null}}));
    """, tmp_path)
    assert reloaded == {"title": "Poll", "ids": [["c", "Route"]], "one": None}


def test_stale_writer_lock_from_killed_sidecar_is_replaced(tmp_path: Path) -> None:
    state = tmp_path / "state" / "poll-seeds.json"
    state.parent.mkdir()
    lock = state.parent / "poll-seeds.json.lock"
    lock.write_text("", encoding="utf-8")
    old = time.time() - 60
    os.utime(lock, (old, old))
    _, stderr = _seed_node(f"""
      const store=new PollSeedStore({json.dumps(str(state))});
      store.remember('p','Q',[{{text:'A',optionIdentifier:'a'}}]);
      console.log(JSON.stringify(null));
    """, tmp_path)
    assert "could not save poll seeds" not in stderr
    assert not lock.exists()
    reloaded, _ = _seed_node(f"""
      const store=new PollSeedStore({json.dumps(str(state))}).load();
      console.log(JSON.stringify(store.has('p')));
    """, tmp_path)
    assert reloaded is True


def test_poll_seed_file_errors_do_not_break_the_sidecar(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("", encoding="utf-8")
    result, stderr = _seed_node(f"""
      const corrupt=new PollSeedStore({json.dumps(str(corrupt))}).load();
      const memory=new PollSeedStore().load();
      memory.remember('p','Q',[{{text:'A',optionIdentifier:'a'}}]);
      const blocked=new PollSeedStore({json.dumps(str(blocked / "seeds.json"))});
      blocked.remember('p','Q',[{{text:'A',optionIdentifier:'a'}}]);
      console.log(JSON.stringify({{corrupt:corrupt.has('p'),memory:memory.has('p'),blocked:blocked.has('p')}}));
    """, tmp_path)
    assert result == {"corrupt": False, "memory": True, "blocked": True}
    assert "WARNING ignoring unreadable poll seed file" in stderr
    assert "WARNING could not save poll seeds" in stderr
