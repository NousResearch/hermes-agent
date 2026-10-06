"""Poll recovery and patch writes fail without losing previously valid state."""
import json
import subprocess
from pathlib import Path

from tests.plugins.platforms.photon.test_spectrum_patch import (
    _PATCHER, _copy_installed_sdk, _patch, _probe, _write_fixture,
)


def test_installed_sdk_cold_seed_survives_remote_failure_and_orders_votes(tmp_path):
    root = _copy_installed_sdk(tmp_path)
    result = _patch(root)
    assert result.returncode == 0, result.stderr
    chunk = root / "node_modules/@spectrum-ts/imessage/dist/index.js"
    with chunk.open("a") as output:
        output.write("\nexport {toPollOptionMessage, sendContent};\n")
    votes = Path("plugins/platforms/photon/sidecar/poll-votes.mjs").resolve().as_uri()
    probe = _probe(chunk, f"""
      const {{PollVoteTracker}}=await import({json.dumps(votes)});
      const file={json.dumps(str(tmp_path / 'seeds.json'))};
      globalThis.__hermesPhotonPollSeeds=new PollSeedStore(file);
      const options=[{{text:'Route',optionIdentifier:'A'}},{{text:'Calendar',optionIdentifier:'B'}}];
      const sender='any;-;+15551234567';
      await sdk.sendContent({{polls:{{create:async()=>({{pollMessageGuid:'p',title:'',options}})}}}},
        sender,sender,{{type:'poll',title:'Which?',options:options.map(o=>({{title:o.text}}))}});
      globalThis.__hermesPhotonPollSeeds=new PollSeedStore(file).load();
      let fetches=0;
      const remote={{polls:{{get:async()=>{{fetches++;throw new Error('offline');}}}}}};
      const cache=new Map(),tracker=new PollVoteTracker(),counts=[],ids=[];
      for(const [type,sequence] of [['voted',11],['unvoted',12],['voted',13],['unvoted',12],['unvoted',10]]){{
        const [msg]=await sdk.toPollOptionMessage(remote,cache,{{type:'poll.changed',
          pollMessageGuid:'p',chatGuid:sender,actor:{{address:sender,service:'iMessage'}},
          occurredAt:new Date(1000),sequence,delta:{{type,optionIdentifier:'A'}}}},'line');
        ids.push(msg.id);
        counts.push(tracker.normalize(msg.content,msg).tally.Route);
      }}
      console.log(JSON.stringify({{fetches,counts,ids}}));
    """)
    assert probe.returncode == 0, probe.stderr
    result = json.loads(probe.stdout)
    assert result["fetches"] == 0
    assert result["counts"] == [1, 0, 1, 1, 1]
    assert result["ids"][0] != result["ids"][2]
    assert result["ids"][1] == result["ids"][3]


def test_patch_rolls_back_all_chunks_after_second_rename_fails(tmp_path):
    first = _write_fixture(tmp_path)
    second = first.with_name("z.js")
    original = first.read_bytes()
    second.write_bytes(original)
    patcher = _PATCHER.resolve().as_uri()
    probe = subprocess.run(["node", "--input-type=module", "-e", f"""
      import fs from 'node:fs';
      import {{patchSpectrumTs}} from {json.dumps(patcher)};
      const rename=fs.renameSync;let calls=0,error;
      fs.renameSync=(...args)=>{{if(++calls===2)throw new Error('second rename failed');return rename(...args);}};
      try{{patchSpectrumTs({json.dumps(str(tmp_path))});}}catch(err){{error=err;}}
      finally{{fs.renameSync=rename;}}
      console.log(JSON.stringify({{message:error?.message,states:error?.patches,
        files:fs.readdirSync({json.dumps(str(first.parent))})}}));
    """], capture_output=True, text=True, check=False)
    assert probe.returncode == 0, probe.stderr
    result = json.loads(probe.stdout)
    assert result["message"] == "second rename failed"
    assert set(result["states"].values()) == {"missing"}
    assert sorted(result["files"]) == ["index.js", "z.js"]
    assert first.read_bytes() == second.read_bytes() == original
