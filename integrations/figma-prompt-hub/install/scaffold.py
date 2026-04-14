#!/usr/bin/env python3
from pathlib import Path
import json, textwrap


root = Path("/home/josef/Projekte/Automation/figma-hermes-prompt-hub")
paths = [
    "mcp", "scripts", "prompts/raw", "prompts/build", "prompts/schema", "evals/cases"
]
for p in paths:
    (root / p).mkdir(parents=True, exist_ok=True)

schema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["id","version","system","developer","user_template","variables","guardrails","few_shots","eval"],
  "properties": {
    "id": {"type":"string"},
    "version": {"type":"string"},
    "system": {"type":"string"},
    "developer": {"type":"string"},
    "user_template": {"type":"string"},
    "variables": {"type":"array","items":{"type":"string"}},
    "guardrails": {"type":"array","items":{"type":"string"}},
    "few_shots": {
      "type":"array",
      "items":{"type":"object","required":["input","output"],"properties":{
        "input":{"type":"string"},"output":{"type":"string"}
      }}
    },
    "eval": {
      "type":"object",
      "required":["must_include","must_not_include"],
      "properties":{
        "must_include":{"type":"array","items":{"type":"string"}},
        "must_not_include":{"type":"array","items":{"type":"string"}}
      }
    }
  }
}
example = {
  "id": "nous-central-v1",
  "version": "0.1.0",
  "system": "Du bist ein präziser Assistent.",
  "developer": "Antwort kurz, korrekt, ohne Floskeln.",
  "user_template": "Kontext: {{context}}\\nAufgabe: {{task}}",
  "variables": ["context","task"],
  "guardrails": ["Keine erfundenen Fakten", "Bei Unsicherheit klar markieren"],
  "few_shots": [{"input":"task=Summarize X","output":"Kurzfassung ..."}],
  "eval": {
    "must_include": ["Kurzfassung"],
    "must_not_include": ["Great question", "As an AI"]
  }
}

validate_py = textwrap.dedent('''\
import json, sys
from pathlib import Path
from jsonschema import validate

root = Path(__file__).resolve().parents[1]
schema = json.loads((root / "prompts/schema/prompt.schema.json").read_text())
target = Path(sys.argv[1])
data = json.loads(target.read_text())
validate(instance=data, schema=schema)
print("OK:", target)
''')

contract = """Figma Layer Contract (Frame: PROMPT/<id>)
00_system
01_developer
02_user_template
10_guardrail_<n>
20_example_in_<n>
21_example_out_<n>
90_eval_must_include_<n>
91_eval_must_not_include_<n>
"""


(root / "prompts/schema/prompt.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
(root / "prompts/raw/nous-central-v1.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
(root / "scripts/validate_prompt.py").write_text(validate_py, encoding="utf-8")
(root / "prompts/figma-layer-contract.txt").write_text(contract, encoding="utf-8")

print("Scaffold erstellt unter:", root)

