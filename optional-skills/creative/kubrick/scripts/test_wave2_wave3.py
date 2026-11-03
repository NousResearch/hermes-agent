#!/usr/bin/env python3
"""Smoke tests for Wave 2 (Forge feedback/evolution/ledgers) and Wave 3 (adapters/QA/operators)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(*args, expect=0):
    proc = subprocess.run([PY, *map(str, args)], cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != expect:
        raise SystemExit(f"command failed {args} (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)

        # --- Wave 2: forge signals ---
        forge_in = ROOT / "references/examples/forge-signals/ledger-before-after.yaml"
        bundle = td / "forge-bundle.yaml"
        run(
            ROOT / "scripts/extract_forge_signals.py",
            "--project-id",
            "forge-demo",
            "--input",
            forge_in,
            "--output",
            bundle,
        )
        run(
            ROOT / "scripts/validate_artifact.py",
            "--artifact",
            bundle,
            "--schema",
            ROOT / "schemas/forge-signal-bundle.schema.yaml",
        )
        bundle_data = yaml.safe_load(bundle.read_text())
        assert bundle_data["authority"]["forge_canonical"] is True
        assert bundle_data["authority"]["automatic_corpus_change_allowed"] is False
        assert bundle_data["signals"]["payoff"]["realized"]
        assert any(p["pattern_id"] == "interface_badge_authority_transfer" for p in bundle_data["pattern_evidence"])

        # --- Wave 2: multi-signal evolution + human review gate ---
        compile_receipt = td / "compile.json"
        compile_receipt.write_text(json.dumps({"status": "COMPILED", "selected_primary": "interface_badge_authority_transfer"}), encoding="utf-8")
        retrieval = td / "retrieval.yaml"
        write(
            retrieval,
            {
                "retrieval_receipt": {
                    "confidence": 0.8,
                    "selected_primary_grammar": "interface_badge_authority_transfer",
                    "selected_supporting_grammars": [],
                }
            },
        )
        use1 = td / "use1.yaml"
        use2 = td / "use2.yaml"
        use3 = td / "use3.yaml"
        for path, outcome, feas, payoff, cultural in [
            (use1, "REJECTED", 0.3, "false", "false"),
            (use2, "REJECTED", 0.2, "false", "true"),
            (use3, "REJECTED", 0.2, "false", "true"),
        ]:
            run(
                ROOT / "scripts/record_pattern_outcome.py",
                "--compile-receipt",
                compile_receipt,
                "--retrieval-receipt",
                retrieval,
                "--project-id",
                "forge-demo",
                "--outcome",
                outcome,
                "--production-feasibility",
                str(feas),
                "--payoff-realized",
                payoff,
                "--cultural-boundary-respected",
                cultural,
                "--output",
                path,
            )
        proposal = td / "proposal.yaml"
        multi = td / "multi.yaml"
        run(
            ROOT / "scripts/propose_pattern_evolution.py",
            "--pattern-id",
            "interface_badge_authority_transfer",
            "--receipt",
            use1,
            "--receipt",
            use2,
            "--receipt",
            use3,
            "--forge-bundle",
            bundle,
            "--output",
            proposal,
            "--receipt-output",
            multi,
        )
        run(
            ROOT / "scripts/validate_artifact.py",
            "--artifact",
            proposal,
            "--schema",
            ROOT / "schemas/pattern-evolution-proposal.schema.yaml",
        )
        run(
            ROOT / "scripts/validate_artifact.py",
            "--artifact",
            multi,
            "--schema",
            ROOT / "schemas/multi-signal-evolution-receipt.schema.yaml",
        )
        proposal_data = yaml.safe_load(proposal.read_text())
        multi_data = yaml.safe_load(multi.read_text())
        assert proposal_data["review"]["automatic_application_allowed"] is False
        assert multi_data["human_review_gate"]["automatic_application_allowed"] is False
        assert multi_data["proposed_changes"]["lifecycle_action"] in {"DEPRECATE", "RETIRE"}
        assert multi_data["human_review_gate"]["required"] is True
        assert multi_data["authority"]["forge_canonical"] is True

        # --- Wave 2: ledger first-class persistence ---
        ledger = td / "ledger.yaml"
        run(ROOT / "scripts/symbolic_ledger.py", "init", "--project-id", "forge-demo", "--out", ledger)
        run(
            ROOT / "scripts/symbolic_ledger.py",
            "mutate",
            "--ledger",
            ledger,
            "--motif-id",
            "cracked-badge",
            "--observed-form",
            "cracked access badge",
            "--state",
            "held by junior",
            "--mutation",
            "ownership transfer",
            "--pattern-link",
            "interface_badge_authority_transfer",
        )
        run(ROOT / "scripts/symbolic_ledger.py", "apply-forge", "--ledger", ledger, "--forge-bundle", bundle)
        snap = td / "retrieval-snap.yaml"
        run(ROOT / "scripts/symbolic_ledger.py", "export-retrieval", "--ledger", ledger, "--out", snap)
        ledger_data = yaml.safe_load(ledger.read_text())
        assert ledger_data.get("pattern_history")
        assert any(h.get("pattern_id") == "interface_badge_authority_transfer" for h in ledger_data["pattern_history"])
        assert "snapshot_hash" in yaml.safe_load(snap.read_text())

        # --- Wave 3: provider adapters share graph identity ---
        graph = {
            "graph_id": "graph-wave3",
            "validation": {"status": "VALID"},
            "nodes": [
                {"id": "badge", "observed_form": "cracked badge", "initial_state": "held", "target_state": "transferred"},
                {"id": "door", "observed_form": "doorway", "initial_state": "closed", "target_state": "open"},
            ],
            "edges": [],
            "convergence_sites": [{"site_id": "primary", "node_ids": ["badge", "door"], "edge_ids": [0], "observable_effect": "access changes", "mask_priority": 0.9}],
            "residue": ["badge scratch remains"],
            "surface_output": {"geometry": ["doorway threshold"], "light": ["dual light"], "material": ["metal badge"]},
        }
        storyboard = {
            "graph_id": "graph-wave3",
            "frames": [
                {
                    "frame_id": "frame-001",
                    "node_states": {"badge": "held", "door": "closed"},
                    "motif_ownership": {"badge": "supervisor"},
                    "object_states": {},
                    "light_states": {"threshold": "balanced"},
                    "material_states": {"badge": "cracked"},
                    "residue": ["badge scratch remains"],
                    "convergence_sites": ["primary"],
                    "prohibited_resets": ["badge ownership without transfer"],
                },
                {
                    "frame_id": "frame-002",
                    "node_states": {"badge": "transferred", "door": "open"},
                    "motif_ownership": {"badge": "junior"},
                    "object_states": {},
                    "light_states": {"threshold": "junior-side-bright"},
                    "material_states": {"badge": "cracked"},
                    "residue": ["badge scratch remains"],
                    "convergence_sites": ["primary"],
                    "prohibited_resets": ["badge ownership without transfer"],
                },
            ],
        }
        graph_path = td / "graph.yaml"
        board_path = td / "board.yaml"
        write(graph_path, graph)
        write(board_path, storyboard)
        for provider in ("flux", "sd3", "midjourney", "grok-imagine"):
            packet = td / f"{provider}-neutral.yaml"
            adapted = td / f"{provider}-adapted.yaml"
            run(
                ROOT / "scripts/build_model_adapter_packet.py",
                "--graph",
                graph_path,
                "--storyboard",
                board_path,
                "--provider",
                provider if provider != "grok-imagine" else "generic",
                "--output",
                packet,
            )
            # force provider field for adapt_provider compatibility when generic
            pdata = yaml.safe_load(packet.read_text())
            pdata["provider"] = "generic"
            write(packet, pdata)
            run(
                ROOT / "scripts/adapt_provider.py",
                "--packet",
                packet,
                "--provider",
                provider,
                "--output",
                adapted,
            )
            adata = yaml.safe_load(adapted.read_text())
            assert adata["validation"]["status"] == "VALID"
            assert adata["source_graph_id"] == "graph-wave3"
            assert adata["shared_latent_graph"]["source_graph_id"] == "graph-wave3"
            assert adata["intent_policy"]["canonical_symbolic_intent_mutable"] is False
            assert len(adata["frames"]) >= 2
            # Audience-facing frame text must not leak private pattern/lexicon semantics or named esoterica.
            frame_blob = json.dumps(adata.get("frames") or []).lower()
            for banned in ("nigredo", "rubedo", "syzygy"):
                assert banned not in frame_blob
            assert '"pattern_links"' not in frame_blob
            assert '"lexicon_links"' not in frame_blob
            assert adata.get("private_state_policy", {}).get("pattern_links_exposed") is False
            assert adata.get("private_state_policy", {}).get("lexicon_links_exposed") is False

        # --- Wave 3: closed-loop QA differential scores ---
        obs = td / "raw-obs.yaml"
        write(
            obs,
            {
                "geometry": ["doorway threshold"],
                "node_states": {"badge": "held", "door": "closed"},
                "motif_ownership": {"badge": "supervisor"},
                "object_states": {},
                "light_states": {"threshold": "balanced"},
                "material_states": {"badge": "cracked"},
                "residue": ["badge scratch remains"],
                "convergence_sites": ["primary"],
            },
        )
        qa_out = td / "qa"
        run(
            ROOT / "scripts/closed_loop_visual_qa.py",
            "--expected",
            board_path,
            "--observation-input",
            obs,
            "--source-graph-id",
            "graph-wave3",
            "--frame-id",
            "frame-001",
            "--out",
            qa_out,
        )
        qa = yaml.safe_load((qa_out / "closed-loop-qa-receipt.yaml").read_text())
        assert qa["status"] == "PASS"
        for key in ("geometry_fidelity", "state_fidelity", "residue_fidelity", "convergence_fidelity"):
            assert key in qa["differential"]
            assert qa["differential"][key]["score"] is not None

        # fail closed on graph mismatch
        bad_obs = td / "bad-obs.yaml"
        write(bad_obs, {"geometry": ["doorway threshold"], "residue": ["badge scratch remains"], "convergence_sites": ["primary"]})
        bad_out = td / "qa-bad"
        proc = subprocess.run(
            [
                PY,
                str(ROOT / "scripts/closed_loop_visual_qa.py"),
                "--expected",
                board_path,
                "--observation-input",
                bad_obs,
                "--source-graph-id",
                "wrong-graph",
                "--frame-id",
                "frame-001",
                "--out",
                str(bad_out),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert proc.returncode != 0
        bad_qa = yaml.safe_load((bad_out / "closed-loop-qa-receipt.yaml").read_text())
        assert bad_qa["status"] == "NOT_COMPUTABLE"

        # --- Wave 3: operators produce receipts ---
        for cmd in (
            ["saturation-score", "--ledger", str(ledger)],
            ["counterpoint", "--packet", str(td / "packet-cp.yaml")],
            ["convergence-lock", "--graph", str(graph_path)],
            ["surface-occult-audit", "--input", str(graph_path)],
            ["symbolic-architecture-export", "--graph", str(graph_path), "--ledger", str(ledger)],
        ):
            if cmd[0] == "counterpoint":
                write(
                    td / "packet-cp.yaml",
                    {
                        "channels": {
                            "diegetic": ["cracked badge changes hands"],
                            "dramaturgical": ["authority reverses at doorway"],
                            "cinematic": ["dual light splits threshold"],
                        }
                    },
                )
            out = td / f"op-{cmd[0]}.yaml"
            run(ROOT / "scripts/graph_operators.py", *cmd, "--output", out)
            data = yaml.safe_load(out.read_text())
            assert data["authority"]["automatic_application_allowed"] is False
            assert data["status"] in {"OK", "PASS", "WARN"}

        occult = td / "occult.yaml"
        write(occult, {"audience_prompt": "show the nigredo stage as black mud"})
        occult_out = td / "occult-receipt.yaml"
        proc = subprocess.run(
            [PY, str(ROOT / "scripts/graph_operators.py"), "surface-occult-audit", "--input", str(occult), "--output", str(occult_out)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert proc.returncode != 0
        assert yaml.safe_load(occult_out.read_text())["status"] == "FAIL"

        # MCP tools/list smoke (one-shot initialize + tools/list)
        mcp = subprocess.run(
            [PY, str(ROOT / "scripts/mcp_kubrick_server.py")],
            input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n",
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert mcp.returncode == 0
        listed = json.loads(mcp.stdout.strip().splitlines()[0])
        names = {t["name"] for t in listed["result"]["tools"]}
        assert "kubrick_do" in names
        # Single primary tool over intent router (legacy multi-tool surface retired)
        assert names == {"kubrick_do"}

    print("wave2/wave3 smoke test: PASS")


if __name__ == "__main__":
    main()
