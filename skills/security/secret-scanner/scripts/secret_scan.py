#!/usr/bin/env python3
"""Secret scanner for Hermes Agent using detect-secrets or trufflehog."""
import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

def run_detect_secrets_scan(path: str, baseline: Optional[str] = None, update: bool = False, ignore_plugins: Optional[List[str]] = None) -> Dict:
    """Run detect-secrets scan."""
    try:
        from detect_secrets import SecretsCollection
        from detect_secrets.settings import transient_settings
    except ImportError:
        print("Error: detect-secrets not installed. Run: pip install detect-secrets", file=sys.stderr)
        sys.exit(1)
    
    plugins_to_ignore = ignore_plugins or []
    
    settings = transient_settings()
    if plugins_to_ignore:
        settings['plugins_used'] = [
            p for p in settings['plugins_used']
            if p['name'] not in plugins_to_ignore
        ]
    
    with settings:
        secrets = SecretsCollection()
        secrets.scan_path(path)
        
        if baseline and Path(baseline).exists():
            # Load baseline and compare
            baseline_secrets = SecretsCollection()
            baseline_secrets.load_baseline(baseline)
            # We'll just return current secrets for simplicity
            # In a real implementation, we'd compute delta
            pass
        
        if update and baseline:
            secrets.save_baseline(baseline)
        
        return secrets.json()

def run_trufflehog_scan(path: str, git_history: bool = False, extra_args: List[str] = None) -> List[Dict]:
    """Run trufflehog scan."""
    try:
        # Check if trufflehog is available
        subprocess.run(['trufflehog', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: trufflehog not installed. Run: pip install trufflehog", file=sys.stderr)
        sys.exit(1)
    
    cmd = ['trufflehog', '--json']
    if git_history:
        cmd.append('--git')
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(path)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse JSON lines
        findings = []
        for line in result.stdout.strip().split('\n'):
            if line:
                findings.append(json.loads(line))
        return findings
    except subprocess.CalledProcessError as e:
        print(f"Trufflehog failed: {e.stderr}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser(description="Scan for secrets in code and configs")
    parser.add_argument("--path", default=".", help="Path to scan (default: .)")
    parser.add_argument("--git-history", action="store_true", help="Scan git history")
    parser.add_argument("--engine", choices=["detect-secrets", "trufflehog"], default="detect-secrets")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline file (detect-secrets only)")
    parser.add_argument("--baseline", default=".secrets.baseline", help="Baseline file path")
    parser.add_argument("--ignore-plugins", help="Comma-separated list of detect-secrets plugins to ignore")
    parser.add_argument("--trufflehog-args", help="Additional arguments for trufflehog")
    args = parser.parse_args()
    
    ignore_plugins = args.ignore_plugins.split(',') if args.ignore_plugins else []
    trufflehog_extra = args.trufflehog_args.split() if args.trufflehog_args else []
    
    if args.engine == "detect-secrets":
        result = run_detect_secrets_scan(
            path=args.path,
            baseline=args.baseline if args.update_baseline else None,
            update=args.update_baseline,
            ignore_plugins=ignore_plugins
        )
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            # Format as table
            results = result.get('results', {})
            if not results:
                print("No secrets found.")
                return
            
            print(f"{'File':<50} {'Line':<6} {'Type':<20} {'Secret'}")
            print("-" * 100)
            for filename, secrets in results.items():
                for secret in secrets:
                    line = secret.get('line_number', 0)
                    typ = secret.get('type', 'Unknown')
                    # Mask the secret for display
                    secret_val = secret.get('hashed_secret', '')[:8] + '...'
                    print(f"{filename:<50} {line:<6} {typ:<20} {secret_val}")
    else:  # trufflehog
        findings = run_trufflehog_scan(
            path=args.path,
            git_history=args.git_history,
            extra_args=trufflehog_extra
        )
        if not findings:
            print("No secrets found.")
            return
        
        if args.format == "json":
            print(json.dumps(findings, indent=2))
        else:
            print(f"{'File':<50} {'Line':<6} {'Type':<20} {'Secret'}")
            print("-" * 100)
            for f in findings:
                file_path = f.get('SourceMetadata', {}).get('Data', {}).get('Filesystem', {}).get('path', '')
                line_no = f.get('SourceMetadata', {}).get('Data', {}).get('Line', 0)
                # trufflehog doesn't always have a clear type, use Reason or DetectorName
                typ = f.get('Reason', f.get('DetectorName', 'Unknown'))
                # Mask the secret
                secret_val = f.get('Raw', '')[:8] + '...' if f.get('Raw') else ''
                print(f"{file_path:<50} {line_no:<6} {typ:<20} {secret_val}")

if __name__ == "__main__":
    main()