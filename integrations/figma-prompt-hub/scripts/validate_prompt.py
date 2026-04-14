import json, sys
from pathlib import Path
from jsonschema import validate

root = Path(__file__).resolve().parents[1]
schema = json.loads((root / "prompts/schema/prompt.schema.json").read_text())
target = Path(sys.argv[1])
data = json.loads(target.read_text())
validate(instance=data, schema=schema)
print("OK:", target)
