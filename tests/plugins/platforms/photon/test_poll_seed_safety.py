"""Seed persistence stays bounded, private, and safe for concurrent writers."""
import json
import subprocess

from tests.plugins.platforms.photon.test_poll_votes import _SEEDS, _seed_node, _node


OPTIONS = "[{text:'Route',optionIdentifier:'A'},{text:'Calendar',optionIdentifier:'B'}]"


def test_oversized_seed_is_skipped_before_json_read(tmp_path):
    file = tmp_path / "seeds.json"
    file.write_text('{"polls":[],"unused":"' + "x" * (2 * 1024 * 1024) + '"}')
    result, warning = _seed_node(f"""
      const store=new PollSeedStore({json.dumps(str(file))}).load();
      console.log(JSON.stringify({{size:store.polls.size}}));
    """, tmp_path)
    assert result == {"size": 0}
    assert "WARNING ignoring unreadable poll seed file: oversized" in warning


def test_seed_write_uses_private_unique_temp_and_merges_other_store(tmp_path):
    file = tmp_path / "seeds.json"
    result, _ = _seed_node(f"""
      import fs from 'node:fs';
      const file={json.dumps(str(file))};
      const temp=`${{file}}.${{process.pid}}.tmp`;
      fs.writeFileSync(temp,'old');fs.chmodSync(temp,0o666);
      const a=new PollSeedStore(file),b=new PollSeedStore(file);
      a.remember('a','Which?',{OPTIONS});b.remember('b','Which?',{OPTIONS});
      console.log(JSON.stringify({{mode:fs.statSync(file).mode&0o777,
        polls:[...new PollSeedStore(file).load().polls.keys()],old:fs.readFileSync(temp,'utf8')}}));
    """, tmp_path)
    assert result == {"mode": 0o600, "polls": ["a", "b"], "old": "old"}


def test_concurrent_seed_processes_preserve_both_polls(tmp_path):
    file = tmp_path / "seeds.json"
    script = tmp_path / "writer.mjs"
    script.write_text(f"""
      import {{PollSeedStore}} from {json.dumps(_SEEDS)};
      const [file,id]=process.argv.slice(2),store=new PollSeedStore(file).load();
      process.stdout.write('ready\\n');
      for await(const _ of process.stdin){{store.remember(id,'Which?',{OPTIONS});break;}}
    """)
    children = [subprocess.Popen(["node", str(script), str(file), name],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True) for name in ["a", "b"]]
    try:
        for child in children:
            assert child.stdout.readline() == "ready\n"
        for child in children:
            child.stdin.write("go\n")
            child.stdin.flush()
        for child in children:
            _, errors = child.communicate(timeout=10)
            assert child.returncode == 0, errors
        assert {poll["id"] for poll in json.loads(file.read_text())["polls"]} == {"a", "b"}
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate()


def test_sent_id_history_survives_seed_eviction_and_missing_identifiers(tmp_path):
    file = tmp_path / "seeds.json"
    result, _ = _seed_node(f"""
      const file={json.dumps(str(file))},store=new PollSeedStore(file,2);
      for(const id of ['a','b','c'])store.remember(id,'Which?',{OPTIONS});
      store.remember('no-identifiers','Which?',[{{text:'Route'}}]);
      const saved=new PollSeedStore(file,2).load();
      const memory=new PollSeedStore();
      for(let i=0;i<2500;i++)memory.remember(`p-${{i}}`,'Which?',[]);
      console.log(JSON.stringify({{polls:[...saved.polls.keys()],ids:[...saved.sentPollIds.keys()],
        retained:memory.sentPollIds.size,old:memory.sentPollIds.has('p-0'),recent:memory.sentPollIds.has('p-2499')}}));
    """, tmp_path)
    assert result == {"polls": ["b", "c"], "ids": ["a", "b", "c", "no-identifiers"],
                      "retained": 2000, "old": False, "recent": True}


def test_evicted_tally_is_partial_when_recreated():
    result = _node("""
      const tracker=new PollVoteTracker(2);tracker.noteCreated('p');
      const vote=(id,voter)=>tracker.normalize({pollId:id,title:'Route',selected:true},
        {id:`${id}:${voter}:A:selected:1`,sender:{id:voter}});
      const first=vote('p','a');vote('q','b');vote('r','b');
      tracker.noteCreated('p');
      const returned=vote('p','c');
      console.log(JSON.stringify({first:first.partial,returned:returned.partial,tally:returned.tally}));
    """)
    assert result == {"first": False, "returned": True, "tally": {"Route": 1}}
