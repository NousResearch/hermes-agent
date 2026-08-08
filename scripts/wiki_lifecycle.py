#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.wiki_lifecycle import WikiLifecycleStore, init_wiki_structure


def cmd_init(args: argparse.Namespace) -> dict:
    return init_wiki_structure(args.wiki_dir, domain=args.domain)


def cmd_add_source(args: argparse.Namespace) -> dict:
    store = WikiLifecycleStore(args.db)
    try:
        source_id = store.add_source(
            source_type=args.source_type,
            title=args.title,
            path=args.path,
            freshness_weight=args.freshness_weight,
        )
        return {"ok": True, "source_id": source_id}
    finally:
        store.close()


def cmd_add_claim(args: argparse.Namespace) -> dict:
    store = WikiLifecycleStore(args.db)
    try:
        claim_id = store.add_claim(
            subject=args.subject,
            predicate=args.predicate,
            object_value=args.object_value,
            claim_text=args.claim_text,
            domain_tag=args.domain_tag,
            volatility_class=args.volatility_class,
            page_path=args.page_path,
        )
        return {"ok": True, "claim_id": claim_id}
    finally:
        store.close()


def cmd_add_evidence(args: argparse.Namespace) -> dict:
    store = WikiLifecycleStore(args.db)
    try:
        evidence_id = store.add_evidence(
            claim_id=args.claim_id,
            source_id=args.source_id,
            evidence_quote=args.evidence_quote,
            evidence_strength=args.evidence_strength,
        )
        return {"ok": True, "evidence_id": evidence_id}
    finally:
        store.close()


def cmd_supersede(args: argparse.Namespace) -> dict:
    store = WikiLifecycleStore(args.db)
    try:
        sid = store.supersede_claim(
            old_claim_id=args.old_claim_id,
            new_claim_id=args.new_claim_id,
            reason=args.reason,
        )
        return {"ok": True, "supersession_id": sid}
    finally:
        store.close()


def cmd_recompute(args: argparse.Namespace) -> dict:
    store = WikiLifecycleStore(args.db)
    try:
        updated = store.recompute_all_confidence(now=datetime.now(timezone.utc))
        return {"ok": True, "updated": updated}
    finally:
        store.close()


def cmd_query(args: argparse.Namespace) -> dict:
    store = WikiLifecycleStore(args.db)
    try:
        items = store.query_claims(args.query, min_confidence=args.min_confidence)
        return {"ok": True, "count": len(items), "items": items}
    finally:
        store.close()


def cmd_lint(args: argparse.Namespace) -> dict:
    store = WikiLifecycleStore(args.db)
    try:
        issues = store.lint(now=datetime.now(timezone.utc))
        return {"ok": True, "issue_count": len(issues), "issues": issues}
    finally:
        store.close()


def cmd_snapshot(args: argparse.Namespace) -> dict:
    store = WikiLifecycleStore(args.db)
    try:
        snap = store.export_snapshot()
        return {"ok": True, **snap}
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wiki lifecycle helper")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_init = sp.add_parser("init")
    p_init.add_argument("--wiki-dir", required=True)
    p_init.add_argument("--domain", default="nasa-upotreba")
    p_init.set_defaults(handler=cmd_init)

    p_src = sp.add_parser("add-source")
    p_src.add_argument("--db", required=True)
    p_src.add_argument("--source-type", default="doc")
    p_src.add_argument("--title", required=True)
    p_src.add_argument("--path", default="")
    p_src.add_argument("--freshness-weight", type=float, default=0.5)
    p_src.set_defaults(handler=cmd_add_source)

    p_claim = sp.add_parser("add-claim")
    p_claim.add_argument("--db", required=True)
    p_claim.add_argument("--subject", required=True)
    p_claim.add_argument("--predicate", required=True)
    p_claim.add_argument("--object-value", required=True)
    p_claim.add_argument("--claim-text", required=True)
    p_claim.add_argument("--domain-tag", default="ops")
    p_claim.add_argument("--volatility-class", default="medium")
    p_claim.add_argument("--page-path", default="")
    p_claim.set_defaults(handler=cmd_add_claim)

    p_evd = sp.add_parser("add-evidence")
    p_evd.add_argument("--db", required=True)
    p_evd.add_argument("--claim-id", required=True)
    p_evd.add_argument("--source-id", required=True)
    p_evd.add_argument("--evidence-quote", required=True)
    p_evd.add_argument("--evidence-strength", type=float, default=0.8)
    p_evd.set_defaults(handler=cmd_add_evidence)

    p_sup = sp.add_parser("supersede")
    p_sup.add_argument("--db", required=True)
    p_sup.add_argument("--old-claim-id", required=True)
    p_sup.add_argument("--new-claim-id", required=True)
    p_sup.add_argument("--reason", default="")
    p_sup.set_defaults(handler=cmd_supersede)

    p_rec = sp.add_parser("recompute")
    p_rec.add_argument("--db", required=True)
    p_rec.set_defaults(handler=cmd_recompute)

    p_q = sp.add_parser("query")
    p_q.add_argument("--db", required=True)
    p_q.add_argument("--query", required=True)
    p_q.add_argument("--min-confidence", type=float, default=0.6)
    p_q.set_defaults(handler=cmd_query)

    p_l = sp.add_parser("lint")
    p_l.add_argument("--db", required=True)
    p_l.set_defaults(handler=cmd_lint)

    p_s = sp.add_parser("snapshot")
    p_s.add_argument("--db", required=True)
    p_s.set_defaults(handler=cmd_snapshot)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = args.handler(args)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
