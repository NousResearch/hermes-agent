from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from institutional_btc_vol.backtest_report import run_manifest_backtest, run_multi_tenor_backtests
from institutional_btc_vol.evidence_verify import verify_evidence_bundle
from institutional_btc_vol.investor_memo import write_investor_memo
from institutional_btc_vol.investor_packet import build_investor_packet
from institutional_btc_vol.investor_site import write_investor_site
from institutional_btc_vol.packet_verify import verify_investor_packet
from institutional_btc_vol.legal_wrapper import write_legal_wrapper_package
from institutional_btc_vol.quote_templates import write_quote_evidence_template
from institutional_btc_vol.site_data import build_site_data, write_site_data
from institutional_btc_vol.source_intake import (
    SOURCE_REQUIREMENTS,
    build_source_manifest_entry,
    load_source_intake_manifest,
    validate_source_intake_manifest,
    write_source_intake_entry,
    write_source_intake_template,
    write_source_intake_validation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btc-vol-monitor",
        description="Internal BTC vol desk monitor utilities. All outputs remain SCREEN-ONLY · NOT EXECUTABLE.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_parser = subcommands.add_parser("run", help="run the live monitor and generate artifacts")
    run_parser.add_argument(
        "base_dir",
        nargs="?",
        default="artifacts/institutional/data",
        help="artifact data directory (default: artifacts/institutional/data)",
    )

    verify_parser = subcommands.add_parser("verify-bundle", help="verify evidence bundle hashes and controls")
    verify_parser.add_argument("bundle", help="path to evidence bundle zip")

    site_parser = subcommands.add_parser("build-site", help="build the investor proof-of-concept site")
    site_parser.add_argument("data_dir", help="artifact data directory with run_manifest.jsonl")
    site_parser.add_argument(
        "output",
        nargs="?",
        default="artifacts/institutional/site/index.html",
        help="output index.html path (default: artifacts/institutional/site/index.html)",
    )

    memo_parser = subcommands.add_parser("build-memo", help="build the investor memo markdown packet")
    memo_parser.add_argument("data_dir", help="artifact data directory with run_manifest.jsonl")
    memo_parser.add_argument(
        "output",
        nargs="?",
        default="artifacts/institutional/investor-packet/btc-vol-desk-investor-memo.md",
        help="output memo path (default: artifacts/institutional/investor-packet/btc-vol-desk-investor-memo.md)",
    )

    packet_parser = subcommands.add_parser("build-packet", help="build complete investor diligence packet")
    packet_parser.add_argument("data_dir", help="artifact data directory with run_manifest.jsonl")
    packet_parser.add_argument(
        "output_dir",
        nargs="?",
        default="artifacts/institutional/investor-packet",
        help="output packet directory (default: artifacts/institutional/investor-packet)",
    )

    verify_packet_parser = subcommands.add_parser("verify-packet", help="verify investor packet hashes, controls, and scans")
    verify_packet_parser.add_argument(
        "packet_dir",
        nargs="?",
        default="artifacts/institutional/investor-packet",
        help="packet directory (default: artifacts/institutional/investor-packet)",
    )

    backtest_parser = subcommands.add_parser("backtest-spread", help="run screen-only point-in-time spread backtest")
    backtest_parser.add_argument("data_dir", help="artifact data directory with run_manifest.jsonl")
    backtest_parser.add_argument(
        "output",
        nargs="?",
        default="artifacts/institutional/backtests/vol-spread-backtest.md",
        help="output markdown path (default: artifacts/institutional/backtests/vol-spread-backtest.md)",
    )
    backtest_parser.add_argument("--tenor", default="7d", help="spread tenor field to backtest, e.g. 7d")
    backtest_parser.add_argument("--threshold-vol-pts", type=float, default=5.0)
    backtest_parser.add_argument("--notional-vega", type=float, default=10_000.0)
    backtest_parser.add_argument("--cost-per-trade", type=float, default=250.0)
    backtest_parser.add_argument("--slippage-vol-pts", type=float, default=0.0)
    backtest_parser.add_argument("--summary-json", help="optional machine-readable summary JSON path")
    backtest_parser.add_argument("--multi-tenor", action="store_true", help="run 1d/7d/30d research scenarios")

    intake_template_parser = subcommands.add_parser("write-source-intake-template", help="write licensed source intake manifest template")
    intake_template_parser.add_argument("output", help="output source_intake_manifest template JSON path")

    intake_validate_parser = subcommands.add_parser("validate-source-intake", help="validate licensed/replay-ready source intake manifest")
    intake_validate_parser.add_argument("manifest", help="source_intake_manifest.json path")
    intake_validate_parser.add_argument("--decision-ts", help="override decision timestamp, ISO-8601")
    intake_validate_parser.add_argument("--output", help="optional markdown validation report path")

    intake_entry_parser = subcommands.add_parser("build-source-intake-entry", help="hash and describe one raw licensed source file")
    intake_entry_parser.add_argument("raw_file", help="raw csv/jsonl/parquet source file path")
    intake_entry_parser.add_argument("--source-group", required=True, choices=list(SOURCE_REQUIREMENTS), help="required source group name")
    intake_entry_parser.add_argument("--provenance", required=True, help="provider/license provenance label")
    intake_entry_parser.add_argument("--license-label", required=True, help="license or contract label")
    intake_entry_parser.add_argument("--source-ref", required=True, help="vendor export id or internal source reference")
    intake_entry_parser.add_argument("--output", required=True, help="output JSON entry path")

    quote_template_parser = subcommands.add_parser("write-quote-evidence-template", help="write manual quote-evidence JSON template")
    quote_template_parser.add_argument("output", help="output quote evidence template JSON path")

    legal_parser = subcommands.add_parser("write-legal-wrapper", help="write counsel-review legal/business wrapper draft")
    legal_parser.add_argument("output_json", help="output legal wrapper JSON path")
    legal_parser.add_argument("--markdown", help="optional markdown output path")

    return parser


def _parse_cli_decision_ts(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify-bundle":
        result = verify_evidence_bundle(Path(args.bundle))
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    if args.command == "run":
        from institutional_btc_vol.run_monitor import main as run_main

        return run_main([args.base_dir])

    if args.command == "build-site":
        output = Path(args.output)
        data = build_site_data(Path(args.data_dir))
        site_path = write_investor_site(data, output)
        site_data_path = write_site_data(Path(args.data_dir), output.with_name("site-data.json"))
        result = {
            "ok": True,
            "site_path": str(site_path),
            "site_data_path": str(site_data_path),
            "run_id": data.get("latest_run", {}).get("run_id"),
            "evidence_status": data.get("evidence_status"),
        }
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "build-memo":
        output = Path(args.output)
        data = build_site_data(Path(args.data_dir))
        memo_path = write_investor_memo(data, output)
        result = {
            "ok": True,
            "memo_path": str(memo_path),
            "run_id": data.get("latest_run", {}).get("run_id"),
            "evidence_status": data.get("evidence_status"),
        }
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "build-packet":
        result = build_investor_packet(Path(args.data_dir), Path(args.output_dir))
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    if args.command == "verify-packet":
        result = verify_investor_packet(Path(args.packet_dir))
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    if args.command == "write-source-intake-template":
        path = write_source_intake_template(Path(args.output))
        print(json.dumps({"ok": True, "template_path": str(path), "evidence_status": "SCREEN-ONLY · SOURCE INTAKE TEMPLATE · NOT EXECUTABLE"}, indent=2))
        return 0

    if args.command == "validate-source-intake":
        manifest = load_source_intake_manifest(Path(args.manifest))
        decision_value = args.decision_ts or manifest.get("decision_ts")
        result = validate_source_intake_manifest(manifest, decision_ts=_parse_cli_decision_ts(decision_value))
        report_path = None
        if args.output:
            report_path = write_source_intake_validation_report(result, Path(args.output))
        payload = {
            "ok": bool(result.get("ready")),
            **result,
            "manifest_path": str(Path(args.manifest)),
        }
        if report_path:
            payload["report_path"] = str(report_path)
        print(json.dumps(payload, indent=2))
        return 0 if result.get("ready") else 1

    if args.command == "build-source-intake-entry":
        entry = build_source_manifest_entry(
            Path(args.raw_file),
            source_group=args.source_group,
            provenance=args.provenance,
            license_label=args.license_label,
            source_ref=args.source_ref,
        )
        entry_path = write_source_intake_entry(entry, Path(args.output))
        payload = {
            "ok": True,
            "entry_path": str(entry_path),
            "source_group": entry.get("source_group"),
            "row_count": entry.get("row_count"),
            "raw_sha256": entry.get("raw_sha256"),
            "evidence_status": "SCREEN-ONLY · SOURCE INTAKE ENTRY · NOT EXECUTABLE",
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "write-quote-evidence-template":
        result = write_quote_evidence_template(Path(args.output))
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "write-legal-wrapper":
        result = write_legal_wrapper_package(Path(args.output_json), Path(args.markdown) if args.markdown else None)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "backtest-spread":
        if args.summary_json or args.multi_tenor:
            result = run_multi_tenor_backtests(
                Path(args.data_dir) / "run_manifest.jsonl",
                Path(args.output),
                Path(args.summary_json) if args.summary_json else Path(args.output).with_suffix(".json"),
                tenors=("1d", "7d", "30d") if args.multi_tenor else (args.tenor,),
                threshold_vol_pts=args.threshold_vol_pts,
                notional_vega=args.notional_vega,
                cost_per_trade=args.cost_per_trade,
                slippage_vol_pts=args.slippage_vol_pts,
            )
            if not args.multi_tenor and result.get("scenarios"):
                first = result["scenarios"][0]
                result = {
                    **result,
                    "trade_count": first.get("trade_count"),
                    "gross_pnl": first.get("gross_pnl"),
                    "max_drawdown": first.get("max_drawdown"),
                    "win_rate": first.get("win_rate"),
                    "snapshot_count": first.get("snapshot_count"),
                }
        else:
            result = run_manifest_backtest(
                Path(args.data_dir) / "run_manifest.jsonl",
                Path(args.output),
                tenor=args.tenor,
                threshold_vol_pts=args.threshold_vol_pts,
                notional_vega=args.notional_vega,
                cost_per_trade=args.cost_per_trade,
            )
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
