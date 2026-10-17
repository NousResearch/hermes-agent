#!/usr/bin/env python3
"""Vérifie que tous les modules s'importent correctement."""
import sys

try:
    from security_recon_assistant.core.models import ScanResult, ScanFinding
    from security_recon_assistant.core.scope import ScopeConfig, load_scope_from_yaml
    from security_recon_assistant.core.guardian import Guardian
    from security_recon_assistant.core.executor import Executor, run
    from security_recon_assistant.scanners.base import BaseScanner
    from security_recon_assistant.scanners.subfinder_scanner import SubfinderScanner
    from security_recon_assistant.scanners.nmap_scanner import NmapScanner
    from security_recon_assistant.orchestrator.pipeline import Pipeline, PipelineConfig
    from security_recon_assistant.reporting.json_report import JSONReportGenerator
    from security_recon_assistant.reporting.html_report import HTMLReportGenerator
    from security_recon_assistant.cli import cli
    print("✅ Tous les imports fonctionnent")
    sys.exit(0)
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
