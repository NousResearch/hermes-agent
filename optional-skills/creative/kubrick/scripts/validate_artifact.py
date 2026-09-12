#!/usr/bin/env python3
"""Validate one Kubrick YAML/JSON artifact against a repository schema."""
from __future__ import annotations
import argparse, json
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")
try:
    from jsonschema import Draft202012Validator
except ImportError:
    raise SystemExit("jsonschema required")


def load(path: Path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text) or {}


def render_path(path) -> str:
    parts=[]
    for item in path:
        parts.append(f"[{item}]" if isinstance(item,int) else (("." if parts else "")+str(item)))
    return "".join(parts) or "$"


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--artifact",required=True)
    p.add_argument("--schema",required=True)
    p.add_argument("--output")
    a=p.parse_args()
    artifact=load(Path(a.artifact)); schema=load(Path(a.schema))
    validator=Draft202012Validator(schema)
    errors=[]
    for error in sorted(validator.iter_errors(artifact), key=lambda e:list(e.absolute_path)):
        errors.append({"path":render_path(error.absolute_path),"message":error.message,"validator":error.validator})
    result={"status":"VALID" if not errors else "INVALID","artifact":a.artifact,"schema":a.schema,"error_count":len(errors),"errors":errors}
    text=json.dumps(result,indent=2)
    if a.output: Path(a.output).write_text(text,encoding="utf-8")
    print(text)
    raise SystemExit(0 if not errors else 1)

if __name__=="__main__": main()
