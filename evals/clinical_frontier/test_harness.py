import json, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class HarnessTest(unittest.TestCase):
    def test_dry_run_and_forbidden_provider(self):
        cp=subprocess.run([sys.executable,str(ROOT/"run_eval.py"),"--dry-run"],capture_output=True,text=True)
        self.assertEqual(cp.returncode,0,cp.stderr); self.assertEqual(json.loads(cp.stdout)["cases"],8)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"models.json"; p.write_text(json.dumps({"providers":[{"id":"claude_bad","enabled":True,"model":"x","command":["anthropic","{prompt_file}"]}]}))
            cp=subprocess.run([sys.executable,str(ROOT/"run_eval.py"),"--models",str(p),"--dry-run"],capture_output=True,text=True)
            self.assertNotEqual(cp.returncode,0)
    def test_fixture_command_preserves_exact_question_and_blinds(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); ds=td/"d.jsonl"; question="¿Texto exacto $(no shell) 38,1 °C?"
            ds.write_text(json.dumps({"id":"c","question":question})+"\n",encoding="utf-8")
            cfg=td/"m.json"; code="import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())"
            cfg.write_text(json.dumps({"providers":[{"id":"fixture","enabled":True,"model":"fixture","command":[sys.executable,"-c",code,"{prompt_file}"]}]}))
            cp=subprocess.run([sys.executable,str(ROOT/"run_eval.py"),"--dataset",str(ds),"--models",str(cfg),"--out",str(td/"runs")],capture_output=True,text=True)
            self.assertEqual(cp.returncode,0,cp.stderr); run=next((td/"runs").iterdir())
            raw=json.loads((run/"responses.jsonl").read_text()); self.assertEqual(raw["response"],question)
            blind=json.loads((run/"blind_responses.jsonl").read_text()); self.assertNotIn("provider_id",blind)
            self.assertFalse(json.loads((run/"manifest.json").read_text())["comparison_complete"])
if __name__=="__main__": unittest.main()
