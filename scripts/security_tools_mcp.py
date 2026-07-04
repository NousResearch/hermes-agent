#!/usr/bin/env python3
"""Security Tools MCP Server — wraps CLI security tools as MCP (Model Context Protocol).

Exposes nmap, nuclei, subfinder, sqlmap, httpx, naabu, ffuf as MCP tools
so any MCP-compatible agent (Hermes, Claude, etc.) can call them.

Transport: stdio (JSON-RPC over stdin/stdout)

Usage:
    python scripts/security_tools_mcp.py

Or register in Hermes MCP config:
    mcp_servers:
      security-tools:
        command: python
        args: [scripts/security_tools_mcp.py]
"""

import json
import subprocess
import sys
import shutil
import os
from typing import Any

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "nmap_scan",
        "description": "Network port scanner — discover open ports and services on a target.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target host or IP (e.g. 'scanme.nmap.org', '192.168.1.1')"},
                "ports": {"type": "string", "description": "Port specification (e.g. '1-1000', '80,443', '--top-ports 100'). Default: top 1000."},
                "flags": {"type": "string", "description": "Additional nmap flags (e.g. '-sV -sC'). Default: '-sV'."},
            },
            "required": ["target"],
        },
    },
    {
        "name": "nuclei_scan",
        "description": "Vulnerability scanner — runs Nuclei templates against a target URL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target URL (e.g. 'https://example.com')"},
                "templates": {"type": "string", "description": "Template tags/paths (e.g. 'cve,misconfig', '/path/to/templates'). Default: all."},
                "severity": {"type": "string", "description": "Filter by severity: info, low, medium, high, critical. Default: all."},
            },
            "required": ["target"],
        },
    },
    {
        "name": "subfinder_enum",
        "description": "Subdomain enumeration — discovers subdomains for a domain using passive sources.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain to enumerate (e.g. 'example.com')"},
                "timeout": {"type": "integer", "description": "Timeout in seconds. Default: 30."},
            },
            "required": ["domain"],
        },
    },
    {
        "name": "sqlmap_scan",
        "description": "SQL injection detection and exploitation tool.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target URL with parameters (e.g. 'http://example.com/page?id=1')"},
                "data": {"type": "string", "description": "POST data (e.g. 'user=admin&pass=test')"},
                "flags": {"type": "string", "description": "Additional sqlmap flags (e.g. '--batch --level=3 --risk=2'). Default: '--batch'."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "httpx_probe",
        "description": "HTTP toolkit — probes URLs for status codes, titles, technologies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Target URL(s) or file path (e.g. 'https://example.com', or '/tmp/hosts.txt')"},
                "flags": {"type": "string", "description": "Additional httpx flags (e.g. '-status-code -title -tech-detect'). Default: '-status-code -title -tech-detect'."},
            },
            "required": ["input"],
        },
    },
    {
        "name": "naabu_scan",
        "description": "Fast port scanner — optimized for speed, covers top ports quickly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Target host or CIDR (e.g. '192.168.1.1', '10.0.0.0/24')"},
                "ports": {"type": "string", "description": "Ports to scan (e.g. 'top-100', '1-65535', '80,443,8080'). Default: 'top-100'."},
            },
            "required": ["host"],
        },
    },
    {
        "name": "ffuf_fuzz",
        "description": "Web fuzzer — directory/file discovery, vhost fuzzing, parameter discovery.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target URL with FUZZ keyword (e.g. 'https://example.com/FUZZ')"},
                "wordlist": {"type": "string", "description": "Wordlist path (e.g. '/usr/share/wordlists/dirb/common.txt'). Default: built-in."},
                "mode": {"type": "string", "description": "Fuzz mode: 'dir' (directories), 'vhost' (virtual hosts), 'param' (parameters). Default: 'dir'."},
            },
            "required": ["url"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 120) -> dict:
    """Run a command and return structured result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout[:50000] if result.stdout else "",
            "stderr": result.stderr[:5000] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except FileNotFoundError:
        return {"exit_code": -1, "stdout": "", "stderr": f"Tool not found: {cmd[0]}"}


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a security tool and return result as text."""

    if name == "nmap_scan":
        target = args["target"]
        ports = args.get("ports", "")
        flags = args.get("flags", "-sV").split()
        cmd = ["nmap"] + flags
        if ports:
            cmd += ["-p", ports]
        cmd += [target]
        result = _run(cmd, timeout=300)
        return result["stdout"] or result["stderr"] or "No output"

    elif name == "nuclei_scan":
        target = args["target"]
        cmd = ["nuclei", "-u", target, "-json"]
        if templates := args.get("templates"):
            cmd += ["-t", templates]
        if severity := args.get("severity"):
            cmd += ["-severity", severity]
        result = _run(cmd, timeout=300)
        return result["stdout"] or result["stderr"] or "No findings"

    elif name == "subfinder_enum":
        domain = args["domain"]
        timeout = args.get("timeout", 30)
        cmd = ["subfinder", "-d", domain, "-silent"]
        result = _run(cmd, timeout=timeout)
        return result["stdout"] or result["stderr"] or "No subdomains found"

    elif name == "sqlmap_scan":
        url = args["url"]
        cmd = ["sqlmap", "-u", url, "--batch"]
        if data := args.get("data"):
            cmd += ["--data", data]
        if flags := args.get("flags", ""):
            cmd += flags.split()
        result = _run(cmd, timeout=300)
        return result["stdout"] or result["stderr"] or "No output"

    elif name == "httpx_probe":
        inp = args["input"]
        flags = args.get("flags", "-status-code -title -tech-detect").split()
        # Use projectdiscovery httpx (Go binary), not Python httpx
        pdx = shutil.which("httpx")
        if pdx and "go/bin" in pdx:
            cmd = [pdx] + flags
        else:
            # Try common Go bin locations
            for candidate in [os.path.expanduser("~/go/bin/httpx"), "/usr/local/go/bin/httpx"]:
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    cmd = [candidate] + flags
                    break
            else:
                return "projectdiscovery httpx not found. Install: go install github.com/projectdiscovery/httpx/cmd/httpx@latest"
        if inp.startswith("/"):
            cmd += ["-l", inp]
        else:
            cmd += ["-u", inp]
        result = _run(cmd, timeout=120)
        return result["stdout"] or result["stderr"] or "No output"

    elif name == "naabu_scan":
        host = args["host"]
        ports = args.get("ports", "top-100")
        cmd = ["naabu", "-host", host]
        if ports.startswith("top-"):
            cmd += ["-top-ports", ports.replace("top-", "")]
        else:
            cmd += ["-p", ports]
        if not shutil.which("naabu"):
            return "naabu not installed. Install: go install github.com/projectdiscovery/naabu/v2/cmd/naabu@latest"
        result = _run(cmd, timeout=120)
        return result["stdout"] or result["stderr"] or "No open ports"

    elif name == "ffuf_fuzz":
        url = args["url"]
        wordlist = args.get("wordlist", "/usr/share/wordlists/dirb/common.txt")
        mode = args.get("mode", "dir")
        if not shutil.which("ffuf"):
            return "ffuf not installed. Install: go install github.com/ffuf/ffuf/v2@latest"
        cmd = ["ffuf", "-u", url, "-w", wordlist, "-mc", "200,204,301,302,307,401,403"]
        if mode == "vhost":
            cmd += ["-H", "Host: FUZZ.target"]
        result = _run(cmd, timeout=300)
        return result["stdout"] or result["stderr"] or "No results"

    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# MCP JSON-RPC protocol (stdio)
# ---------------------------------------------------------------------------

def handle_request(req: dict) -> dict | None:
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "security-tools-mcp", "version": "1.0.0"},
            },
        }

    elif method == "notifications/initialized":
        return None  # notification — no response

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        output = execute_tool(tool_name, tool_args)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": output}],
            },
        }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main():
    """Main stdio loop — read JSON-RPC from stdin, write to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
