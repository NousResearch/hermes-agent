"""Execute the sidecar's poll identity and tally logic under Node."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


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
