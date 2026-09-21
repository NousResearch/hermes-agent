"""Build an inert, self-contained X runtime candidate; never activate cron jobs."""
import hashlib
import json
import shutil
from pathlib import Path


def build_bundle(scripts_dir):
    scripts_dir=Path(scripts_dir)
    if scripts_dir.exists() and any(scripts_dir.iterdir()):
        raise ValueError('candidate scripts directory must be empty')
    scripts_dir.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[2]
    engine=scripts_dir/'content_engine'
    engine.mkdir()
    for source in (repo/'content_engine').glob('*.py'):
        shutil.copy2(source,engine/source.name)
    shutil.copy2(repo/'content_engine/x_voice_runtime.md', engine/'x_voice_runtime.md')
    for source in Path(__file__).parent.glob('x_*.py'):
        if source.name!='x_runtime_bundle.py':
            shutil.copy2(source,scripts_dir/source.name)
    manifest={str(p.relative_to(scripts_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts_dir.rglob('*') if p.is_file()}
    (scripts_dir/'x-runtime-manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('candidate_scripts_dir')
    args=parser.parse_args()
    print(json.dumps(build_bundle(args.candidate_scripts_dir),indent=2))
