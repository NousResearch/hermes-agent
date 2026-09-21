"""Offline analytics importer/report CLI. Outputs only to an explicitly new directory."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here/'content_engine' if (_here/'content_engine').is_dir() else _here.parents[1]/'content_engine'))
from x_analytics import import_csv, import_historical, validate, growth_report, render_growth, voice_export, merge_observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--format', choices=('json','csv','historical'), default='json')
    parser.add_argument('--account', required=True)
    parser.add_argument('--definitions')
    parser.add_argument('--baseline', help='Previous observation JSON to append/dedupe, never overwritten')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.format == 'csv':
        if not args.definitions:
            parser.error('--definitions required for CSV')
        data = import_csv(args.input, args.account, args.definitions)
    elif args.format == 'historical':
        data = import_historical(args.input,args.account,datetime.now(timezone.utc).isoformat())
    else:
        data = validate(json.loads(Path(args.input).read_text()))
    if data['account'].lower() != args.account.lower():
        parser.error('input account differs')
    if args.baseline:
        data = merge_observations(json.loads(Path(args.baseline).read_text()), data)
    report = growth_report(data)
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    for name, value in [('observations',data),('growth-report',report),('voice-reference',voice_export(data))]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2,ensure_ascii=False))
    (output/'growth-report.html').write_text('<!doctype html><meta charset="utf-8">'+render_growth(report))
    print(json.dumps({'output':str(output.resolve()),'unique_posts':report['unique_posts'], 'observations':report['observations']}))


if __name__ == '__main__':
    main()
