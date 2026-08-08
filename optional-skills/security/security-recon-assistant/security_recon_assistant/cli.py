#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import click

from .core.executor import Executor
from .core.guardian import Guardian, ViolationError
from .core.scope import load_scope_from_yaml
from .orchestrator.pipeline import Pipeline, PipelineConfig
from .reporting.html_report import HTMLReportGenerator
from .reporting.json_report import JSONReportGenerator

from .reporting.markdown_report import MarkdownReportGenerator
import shutil
import sys

from .scanners.nmap_scanner import NmapScanner
from .scanners.subfinder_scanner import SubfinderScanner


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--target", "targets", multiple=True, required=False, help="Target domain/IP (repeatable).")
@click.option("--scope", type=click.Path(exists=False, dir_okay=False, path_type=Path), default=Path("scope.yaml"), show_default=True, help="Path to scope YAML file.")
@click.option("--output-format", type=click.Choice(["json", "html", "md"], case_sensitive=False), default="json", show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Output report path.")
@click.option("--workers", type=click.IntRange(1, 64), default=1, show_default=True)
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO", show_default=True)
@click.option("--verbose", is_flag=True, default=False, help="Verbose output.")
@click.option("--quiet", is_flag=True, default=False, help="Minimal output.")
@click.option("--check-deps", is_flag=True, default=False, help="Check if required binaries are installed and exit.")
@click.version_option(version="0.1.0", prog_name="security-recon-assistant")
def cli(targets, scope, output_format, output, workers, log_level, verbose, quiet, check_deps):
    """Security-recon-assistant CLI.

    Professional, scope-guarded reconnaissance orchestration.
    """
    
    if check_deps:
        click.echo("Running Dependency Check for AI Agent Context...")
        tools = ["nmap", "subfinder", "nuclei", "ffuf", "whatweb", "gowitness", "sslscan"]
        all_ok = True
        for t in tools:
            path = shutil.which(t)
            if path:
                click.echo(f"✅ {t}: Installed ({path})")
            else:
                click.echo(f"❌ {t}: NOT FOUND in PATH")
                all_ok = False
        click.echo("")
        if all_ok:
            click.echo("All tools are available!")
            sys.exit(0)
        else:
            click.echo("Some tools are missing. Scans relying on them will fail.")
            sys.exit(1)

    
    if not check_deps and not targets:
        click.echo("Error: Missing option '--target'. Required unless --check-deps is used.")
        sys.exit(2)

    start_time = time.perf_counter()

    try:
        scope_config = load_scope_from_yaml(str(scope))
        guardian = Guardian(scope_config)
        executor = Executor(workers=workers)

        pipeline = Pipeline(
            scanners=[SubfinderScanner(), NmapScanner()],
            guardian=guardian,
            executor=executor,
            config=PipelineConfig(sequential=(workers == 1), retry_failed=False, max_retries=1, stop_on_critical=True),
        )

        all_results = []
        for target in targets:
            if not guardian.is_allowed(target):
                raise ViolationError(f"Target out of scope: {target}")
            all_results.extend(pipeline.run(target))

        duration = time.perf_counter() - start_time

        if output is None:
            output = Path("report.json" if output_format.lower() == "json" else "report.html")

        if output_format.lower() == "json":
            JSONReportGenerator(pretty=True).save(
                filepath=str(output),
                target=targets[0],
                results=all_results,
                scope=scope_config,
                total_duration=duration,
                custom_metadata={"log_level": log_level.upper()},
            )
        else:
            HTMLReportGenerator().save(
                filepath=str(output),
                target=targets[0],
                results=all_results,
                scope=scope_config,
            )

        if not quiet:
            click.echo(f"Report generated: {output}")
            click.echo(f"Scans executed: {len(all_results)}")
            if verbose:
                for result in all_results:
                    click.echo(f"- {result.scanner_name}: success={result.success}, command={result.command}")

    except KeyboardInterrupt:
        click.echo("Interrupted by user.")
        raise SystemExit(130)
    except ViolationError as exc:
        click.echo(f"Scope violation: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}")
        raise SystemExit(1)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
