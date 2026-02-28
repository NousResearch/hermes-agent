#!/usr/bin/env python3
"""terraform_manager — Inspect Terraform workspaces, state, plan output, and outputs.

Usage:
    python terraform_manager.py status              # workspace + state summary
    python terraform_manager.py resources           # list all resources in state
    python terraform_manager.py resources --type aws_instance  # filter by type
    python terraform_manager.py outputs             # show all output values
    python terraform_manager.py workspaces          # list workspaces and active one
    python terraform_manager.py validate            # validate + fmt check
    python terraform_manager.py costs               # resource count breakdown by type
    python terraform_manager.py plan-summary <planfile>  # summarize a saved plan file

No dependencies beyond Python stdlib and terraform in PATH.
Must be run from a Terraform working directory.
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(args: list[str], check: bool = True, capture_stderr: bool = False) -> str:
    """Run a terraform command and return stdout."""
    try:
        result = subprocess.run(
            ["terraform"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
        )
        if capture_stderr and result.stderr:
            return (result.stdout + "\n" + result.stderr).strip()
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        if stderr:
            print(f"Error: {stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: terraform is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)


def _json_cmd(args: list[str]) -> dict | list:
    """Run terraform command with -json flag and parse output."""
    raw = _run(args + ["-json"], check=False)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _check_tf_dir() -> None:
    """Verify we're in a Terraform working directory."""
    tf_files = list(Path(".").glob("*.tf"))
    if not tf_files and not Path(".terraform").exists():
        print("Warning: no .tf files found. Run from a Terraform working directory.", file=sys.stderr)


def _tf_version() -> str:
    """Get installed Terraform version."""
    raw = _run(["version", "-json"], check=False)
    if raw:
        try:
            data = json.loads(raw)
            return data.get("terraform_version", "unknown")
        except json.JSONDecodeError:
            pass
    return _run(["version"], check=False).splitlines()[0] if _run(["version"], check=False) else "unknown"


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_status() -> None:
    """Workspace + state summary."""
    _check_tf_dir()
    print("=== Terraform Status ===\n")

    # Version
    print(f"Version:     {_tf_version()}")

    # Working directory
    print(f"Directory:   {Path.cwd()}")

    # .tf files
    tf_files = list(Path(".").glob("*.tf"))
    print(f"Config files: {len(tf_files)} .tf file(s): {', '.join(f.name for f in tf_files)}")

    # Backend type (read from .terraform/terraform.tfstate)
    backend_file = Path(".terraform") / "terraform.tfstate"
    if backend_file.exists():
        try:
            meta = json.loads(backend_file.read_text())
            backend = meta.get("backend", {}).get("type", "local")
            print(f"Backend:     {backend}")
        except Exception:
            print("Backend:     unknown")
    else:
        print("Backend:     not initialized (run terraform init)")

    # Current workspace
    workspace = _run(["workspace", "show"], check=False)
    print(f"Workspace:   {workspace or 'default'}")

    # State summary
    state_raw = _run(["state", "list"], check=False)
    resources = [r for r in state_raw.splitlines() if r.strip()]
    print(f"Resources:   {len(resources)} in state")

    # Outputs
    outputs_raw = _json_cmd(["output"])
    if isinstance(outputs_raw, dict):
        print(f"Outputs:     {len(outputs_raw)} defined")

    # Lock file
    lock_file = Path(".terraform.lock.hcl")
    print(f"Lock file:   {'present' if lock_file.exists() else 'missing (run terraform init)'}")

    # State file (local only)
    state_file = Path("terraform.tfstate")
    if state_file.exists():
        size = state_file.stat().st_size
        print(f"State file:  local ({size:,} bytes)")

    # tfvars files
    tfvars = list(Path(".").glob("*.tfvars")) + list(Path(".").glob("*.tfvars.json"))
    if tfvars:
        print(f"Var files:   {', '.join(f.name for f in tfvars)}")


def cmd_resources(filter_type: str | None = None) -> None:
    """List all resources tracked in state."""
    _check_tf_dir()

    state_raw = _run(["state", "list"], check=False)
    resources = [r for r in state_raw.splitlines() if r.strip()]

    if not resources:
        print("No resources in state. Run 'terraform apply' first.")
        return

    if filter_type:
        resources = [r for r in resources if filter_type in r]
        if not resources:
            print(f"No resources matching type '{filter_type}'.")
            return

    # Group by provider/type
    grouped: dict[str, list[str]] = {}
    for res in resources:
        # Address format: module.name.type.name or type.name
        parts = res.split(".")
        # Find the resource type (last segment before final name)
        if len(parts) >= 2:
            # Handle module paths like module.vpc.aws_subnet.private
            res_type = parts[-2] if not parts[-2].startswith("module") else parts[-1]
        else:
            res_type = parts[0]

        grouped.setdefault(res_type, []).append(res)

    print(f"Resources in state: {len(resources)}\n")
    print(f"{'ADDRESS':<60} {'TYPE'}")
    print("-" * 80)

    for res in sorted(resources):
        parts = res.split(".")
        res_type = parts[-2] if len(parts) >= 2 else "unknown"
        print(f"{res:<60} {res_type}")

    print(f"\nTotal: {len(resources)} resource(s)")
    if filter_type:
        print(f"Filter: '{filter_type}'")


def cmd_outputs() -> None:
    """Show all Terraform output values."""
    _check_tf_dir()

    data = _json_cmd(["output"])

    if not data or not isinstance(data, dict):
        print("No outputs defined or terraform not initialized.")
        return

    print(f"Outputs ({len(data)}):\n")

    for name, info in sorted(data.items()):
        value     = info.get("value")
        sensitive = info.get("sensitive", False)
        otype     = info.get("type", "unknown")

        print(f"  {name}")
        print(f"    Type:      {otype}")
        if sensitive:
            print(f"    Value:     (sensitive — run: terraform output -raw {name})")
        elif isinstance(value, (dict, list)):
            print(f"    Value:     {json.dumps(value, indent=6)[:200]}")
        else:
            print(f"    Value:     {value}")
        print()


def cmd_workspaces() -> None:
    """List all workspaces and highlight the active one."""
    _check_tf_dir()

    raw = _run(["workspace", "list"], check=False)
    if not raw:
        print("Could not list workspaces. Run 'terraform init' first.")
        return

    current = _run(["workspace", "show"], check=False).strip()

    print(f"Workspaces (current: {current}):\n")
    for line in raw.splitlines():
        line = line.strip().lstrip("* ")
        marker = " ◀ active" if line == current else ""
        print(f"  {line}{marker}")

    print()
    print("Switch:  terraform workspace select <name>")
    print("Create:  terraform workspace new <name>")
    print("Delete:  terraform workspace delete <name>")


def cmd_validate() -> None:
    """Run terraform validate and check formatting."""
    _check_tf_dir()

    print("=== Terraform Validate ===\n")

    # fmt check
    fmt_result = _run(["fmt", "-check", "-recursive"], check=False)
    if fmt_result:
        print("Format check: FAILED — these files need formatting:")
        for line in fmt_result.splitlines():
            print(f"  {line}")
        print("  Fix with: terraform fmt -recursive")
    else:
        print("Format check: OK")

    # validate
    val_raw = _run(["validate", "-json"], check=False)
    try:
        val_data = json.loads(val_raw)
    except json.JSONDecodeError:
        print(f"Validate:     {val_raw[:200]}")
        return

    valid   = val_data.get("valid", False)
    errors  = val_data.get("error_count", 0)
    warns   = val_data.get("warning_count", 0)
    diags   = val_data.get("diagnostics", [])

    print(f"Validate:     {'OK' if valid else 'FAILED'} ({errors} error(s), {warns} warning(s))")

    for d in diags:
        severity = d.get("severity", "error").upper()
        summary  = d.get("summary", "")
        detail   = d.get("detail", "")
        rng      = d.get("range", {})
        fname    = rng.get("filename", "")
        start    = rng.get("start", {})
        line     = start.get("line", "")

        loc = f" [{fname}:{line}]" if fname else ""
        print(f"\n  [{severity}]{loc} {summary}")
        if detail:
            print(f"  {detail[:200]}")


def cmd_costs() -> None:
    """Resource count breakdown by provider and type."""
    _check_tf_dir()

    state_raw = _run(["state", "list"], check=False)
    resources = [r for r in state_raw.splitlines() if r.strip()]

    if not resources:
        print("No resources in state.")
        return

    # Extract provider and type from resource addresses
    # Formats: aws_instance.web, module.vpc.aws_subnet.private[0]
    type_counts: Counter = Counter()
    provider_counts: Counter = Counter()

    for res in resources:
        # Strip module prefix and index suffix
        clean = res
        while clean.startswith("module."):
            parts = clean.split(".", 2)
            clean = parts[2] if len(parts) > 2 else clean

        # Strip index like [0]
        if "[" in clean:
            clean = clean[:clean.index("[")]

        parts = clean.split(".")
        if len(parts) >= 2:
            res_type = parts[-2]
            provider = res_type.split("_")[0] if "_" in res_type else res_type
            type_counts[res_type] += 1
            provider_counts[provider] += 1

    print(f"Resource Cost Estimate — {len(resources)} total resource(s)\n")

    print("By Provider:")
    for provider, count in provider_counts.most_common():
        bar = "█" * min(count, 40)
        print(f"  {provider:<15} {count:>4}  {bar}")

    print("\nBy Type (top 20):")
    print(f"  {'TYPE':<45} {'COUNT'}")
    print("  " + "-" * 55)
    for res_type, count in type_counts.most_common(20):
        print(f"  {res_type:<45} {count}")

    if len(type_counts) > 20:
        print(f"  ... and {len(type_counts) - 20} more types")

    print(f"\nNote: actual cost depends on usage. Use Infracost for pricing estimates.")


def cmd_plan_summary(plan_file: str) -> None:
    """Parse and summarize a saved binary plan file."""
    if not Path(plan_file).exists():
        print(f"Plan file not found: {plan_file}")
        print("Save a plan with: terraform plan -out=tfplan")
        sys.exit(1)

    # Convert binary plan to JSON
    raw = _run(["show", "-json", plan_file], check=False)
    if not raw:
        print("Could not parse plan file. Make sure it was created by 'terraform plan -out=...'")
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("Plan file is not valid JSON output.")
        sys.exit(1)

    changes = data.get("resource_changes", [])

    to_create  = [c for c in changes if "create"  in c.get("change", {}).get("actions", [])]
    to_update  = [c for c in changes if "update"  in c.get("change", {}).get("actions", [])]
    to_delete  = [c for c in changes if "delete"  in c.get("change", {}).get("actions", [])]
    to_replace = [c for c in changes if "create" in c.get("change", {}).get("actions", [])
                  and "delete" in c.get("change", {}).get("actions", [])]
    no_change  = [c for c in changes if c.get("change", {}).get("actions", []) == ["no-op"]]

    print(f"=== Plan Summary: {plan_file} ===\n")
    print(f"  + Create:  {len(to_create)}")
    print(f"  ~ Update:  {len(to_update)}")
    print(f"  - Destroy: {len(to_delete)}")
    print(f"  ± Replace: {len(to_replace)}")
    print(f"  = No-op:   {len(no_change)}")
    print()

    if to_create:
        print("Resources to CREATE:")
        for c in to_create:
            print(f"  + {c.get('address', '')}  [{c.get('type', '')}]")

    if to_update:
        print("\nResources to UPDATE:")
        for c in to_update:
            print(f"  ~ {c.get('address', '')}  [{c.get('type', '')}]")

    if to_delete:
        print("\nResources to DESTROY:")
        for c in to_delete:
            print(f"  - {c.get('address', '')}  [{c.get('type', '')}]")

    if to_replace:
        print("\nResources to REPLACE (destroy + create):")
        for c in to_replace:
            print(f"  ± {c.get('address', '')}  [{c.get('type', '')}]")

    # Terraform version used to create this plan
    tf_meta = data.get("terraform_version", "")
    if tf_meta:
        print(f"\nPlan created with Terraform {tf_meta}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def _get_flag(args: list[str], flag: str) -> str | None:
    """Extract a named flag value from args list."""
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return None


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    cmd  = args[0]
    rest = args[1:]

    if cmd == "status":
        cmd_status()

    elif cmd == "resources":
        filter_type = _get_flag(rest, "--type")
        cmd_resources(filter_type=filter_type)

    elif cmd == "outputs":
        cmd_outputs()

    elif cmd == "workspaces":
        cmd_workspaces()

    elif cmd == "validate":
        cmd_validate()

    elif cmd == "costs":
        cmd_costs()

    elif cmd == "plan-summary":
        if not rest or rest[0].startswith("-"):
            print("Usage: terraform_manager.py plan-summary <planfile>")
            print("       Save a plan with: terraform plan -out=tfplan")
            sys.exit(1)
        cmd_plan_summary(rest[0])

    else:
        print(f"Unknown command: '{cmd}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
