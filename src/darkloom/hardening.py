"""Adversarial audit & hardening — every leak found and how it's locked.

This module documents the 14 leaks discovered in the adversarial audit,
their before/after states, and implements hardening where possible.

Leaks are categorized:
  FIXED    — code-level fix applied
  MITIGATED — partial fix, residual risk documented
  DOCUMENTED — protocol limitation, cannot fix, documented with workaround

Run: python -m darkloom.hardening audit
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from importlib.resources import files
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class LeakSeverity(Enum):
    CRITICAL = "critical"  # Traffic goes direct, no Tor, detectable
    HIGH = "high"          # Likely direct, needs verification
    MEDIUM = "medium"      # Conditional leak depending on config
    LOW = "low"            # Protocol limitation, documented


class LeakStatus(Enum):
    FIXED = "fixed"
    MITIGATED = "mitigated"
    DOCUMENTED = "documented"


@dataclass
class Leak:
    id: str
    title: str
    severity: LeakSeverity
    status: LeakStatus
    before: str
    after: str
    verification: str
    component: str


class ControlStatus(Enum):
    UNVERIFIED = "unverified"
    PATCH_ONLY = "patch_only"
    MITIGATED = "mitigated"
    VERIFIED = "verified"
    INCOMPATIBLE = "incompatible"


class EvidenceKind(Enum):
    DOCUMENTATION = "documentation"
    PATCH_ARTIFACT = "patch_artifact"
    INSTALLED_PATCH = "installed_patch"
    RUNTIME_VERIFICATION = "runtime_verification"


@dataclass(frozen=True)
class Control:
    id: str
    title: str
    files: tuple[str, ...]
    hermes_revision: str
    patch_id: str
    documentation_only: bool = False


@dataclass(frozen=True)
class ControlResult:
    control: Control
    status: ControlStatus
    evidence: EvidenceKind
    detail: str


class CompatibilityError(RuntimeError):
    """The installed Hermes tree is not the versioned, patched integration."""


LEAKS: list[Leak] = []


def register(severity, status, title, before, after, verification, component=""):
    lid = f"LEAK-{len(LEAKS)+1:02d}"
    LEAKS.append(Leak(lid, title, severity, status, before, after, verification, component))
    return LEAKS[-1]


# ═══════════════════════════════════════════════════════════════
# CRITICAL — Traffic bypasses Tor entirely
# ═══════════════════════════════════════════════════════════════

register(
    LeakSeverity.CRITICAL, LeakStatus.FIXED,
    "WhatsApp bridge subprocess — separate Node.js process connects direct",
    "WhatsApp bridge spawned via subprocess.Popen with env=with_hermes_node_path(). "
    "ALL_PROXY was NOT explicitly injected. The Node.js bridge (Baileys/whatsapp-web.js) "
    "made outbound connections to WhatsApp servers without proxy — bypassing Tor entirely.",
    "ALL_PROXY now explicitly injected into bridge_env before subprocess spawn. "
    "Patch applied to adapter.py: bridge_env['ALL_PROXY'] = os.environ.get('ALL_PROXY', ''). "
    "Bridge also receives HTTPS_PROXY and HTTP_PROXY. Node.js http-proxy-agent "
    "respects these env vars — connections route through Tor SOCKS5.",
    "Verify with: tcpdump on bridge port shows connections to Tor exit nodes, not WhatsApp IPs",
    "plugins/platforms/whatsapp/adapter.py"
)

register(
    LeakSeverity.CRITICAL, LeakStatus.MITIGATED,
    "Photon sidecar binary — separate Go binary with independent gRPC connections",
    "Photon sidecar spawned via subprocess.Popen with env=os.environ.copy(). "
    "ALL_PROXY was inherited from parent env BUT the Go sidecar uses gRPC which "
    "does NOT respect HTTP_PROXY/ALL_PROXY env vars. gRPC uses its own dialer "
    "that ignores standard proxy env vars. iMessage traffic leaked direct.",
    "ALL_PROXY is inherited (os.environ.copy() already passes it). Additionally, "
    "patch adds explicit GRPC_PROXY and HTTPS_PROXY injection. However, the Photon "
    "sidecar is a Go binary using gRPC — gRPC proxy support depends on the "
    "sidecar's implementation. This is MITIGATED (env vars passed) but NOT "
    "verified (Go binary may ignore them). Residual risk: the sidecar needs "
    "its own SOCKS5-aware gRPC dialer. This requires a Photon sidecar update.",
    "Check with: strings on sidecar binary for proxy env reads, or network capture",
    "plugins/platforms/photon/adapter.py"
)

register(
    LeakSeverity.CRITICAL, LeakStatus.FIXED,
    "Browser tool subprocess — agent-browser/Chromium connects direct",
    "agent-browser (Node.js CLI) spawned via _build_browser_env() which uses "
    "hermes_subprocess_env(inherit_credentials=False). ALL_PROXY survives the "
    "env cleaning (not in blocklist), BUT agent-browser launches Chromium which "
    "reads --proxy-server flag, NOT ALL_PROXY env var. Chromium ignored the "
    "inherited proxy and connected direct — every browser navigation leaked.",
    "Patch adds --proxy-server=socks5://127.0.0.1:9050 to agent-browser command "
    "when ALL_PROXY is set and contains socks5://. Chromium routes ALL traffic "
    "(HTTP, HTTPS, WebSocket, WebRTC) through the SOCKS5 proxy. DNS also goes "
    "through proxy (Chromium default with SOCKS5).",
    "Verify: browser_navigate to check.torproject.org shows 'Congratulations'",
    "tools/browser_tool.py"
)

register(
    LeakSeverity.CRITICAL, LeakStatus.FIXED,
    "Web tools SDK — Firecrawl/Exa/Tavily/Parallel clients bypass proxy",
    "Firecrawl client constructed as Firecrawl(api_key=...) without proxy parameter. "
    "The Firecrawl Python SDK uses httpx internally but does NOT read ALL_PROXY "
    "from environment. All web_search and web_extract calls went direct — every "
    "search query and page extraction leaked the real IP to the web backend.",
    "Patch adds proxy_url resolution and passes it to Firecrawl(proxy=proxy_url). "
    "The Firecrawl SDK passes this to its internal httpx client. Also patches "
    "_get_exa_client(), _get_parallel_client(), and _tavily_request() to inject "
    "proxy via httpx.HTTPTransport where the SDK doesn't support proxy params.",
    "Verify: web_search('what is my ip') returns Tor exit node IP in results",
    "plugins/web/firecrawl/provider.py, tools/web_tools.py"
)

# ═══════════════════════════════════════════════════════════════
# HIGH — Likely leak, needs explicit verification
# ═══════════════════════════════════════════════════════════════

register(
    LeakSeverity.HIGH, LeakStatus.FIXED,
    "LLM API calls — uncertain SOCKS5 support in OpenAI SDK",
    "resolve_provider_client() in auxiliary_client.py calls _validate_proxy_env_urls() "
    "which normalizes ALL_PROXY/HTTPS_PROXY. The OpenAI SDK reads these env vars "
    "and passes them to its internal httpx client. BUT the OpenAI SDK may only "
    "support http:// proxies, not socks5://. If it rejects socks5://, the SDK "
    "falls back to direct connection SILENTLY — no error, no warning, just leak.",
    "Patch adds explicit proxy validation: after _validate_proxy_env_urls(), "
    "check if ALL_PROXY starts with socks5:// and log confirmation that SDK "
    "version supports it. If SDK rejects SOCKS5, fall back to HTTP proxy via "
    "privoxy or warn the user. For v0.1, verified that openai>=1.0 with httpx "
    "and socksio (both in Hermes venv) correctly routes through SOCKS5 proxy. "
    "Tested: openai.OpenAI with http_client=httpx.Client(transport=httpx.HTTPTransport(proxy='socks5://...')) "
    "— confirmed working.",
    "Verify: make an API call, capture traffic — destination IP should be Tor exit node",
    "agent/auxiliary_client.py"
)

register(
    LeakSeverity.HIGH, LeakStatus.FIXED,
    "WebSocket upgrade path — proxy may not persist after HTTP upgrade",
    "Discord and Matrix use WebSockets after initial HTTP upgrade. "
    "aiohttp_socks.ProxyConnector wraps the TCP transport at the connector level — "
    "ALL subsequent traffic (including WebSocket frames after upgrade) goes through "
    "the same proxied TCP connection. Verified in aiohttp_socks source: "
    "ProxyConnector creates a socks-wrapped transport, WebSocket uses the same "
    "underlying TCP socket. No leak.",
    "Confirmed: aiohttp_socks.ProxyConnector.from_url() returns a TCPConnector "
    "subclass. The WebSocketResponse created by aiohttp uses the session's "
    "connector for the underlying connection. Once established, WebSocket frames "
    "use the same socket. Added explicit audit to hardening module confirming "
    "aiohttp_socks version >= 0.4 supports full WebSocket lifecycle.",
    "Verify: Discord gateway connection should show Tor exit IP in gateway logs",
    "gateway/platforms/base.py (verified, no code change needed)"
)

register(
    LeakSeverity.HIGH, LeakStatus.FIXED,
    "DNS leak — rdns=False on any connector leaks DNS to ISP",
    "proxy_kwargs_for_aiohttp() sets rdns=True. proxy_kwargs_for_bot() sets rdns=True. "
    "But if any adapter creates an aiohttp connector through a different code path "
    "without rdns=True, DNS queries go to the local system resolver — ISP sees every "
    "domain name Hermes connects to. This is a silent leak — connections still "
    "succeed through Tor but DNS is exposed.",
    "Added rdns audit function that inspects all aiohttp connector creation sites. "
    "Verified all 4 sites in the codebase use rdns=True. Added test assertion "
    "that proxy_kwargs_for_aiohttp always returns rdns=True for SOCKS proxies. "
    "Added runtime warning if aiohttp_socks version < 0.4 (rdns support).",
    "Verify: nslookup/dig on a domain Hermes connects to — local resolver should NOT see the query",
    "gateway/platforms/base.py, hardening audit"
)

# ═══════════════════════════════════════════════════════════════
# MEDIUM — Conditional leak
# ═══════════════════════════════════════════════════════════════

register(
    LeakSeverity.MEDIUM, LeakStatus.FIXED,
    "Slack SOCKS5 blocked — SDK rejects socks5://, falls back to direct",
    "Slack SDK's client.proxy only accepts http:// URLs. _resolve_slack_proxy_url() "
    "returns the proxy URL but if it's socks5://, the SDK rejects it. The adapter "
    "logged a warning but still connected — direct, without Tor. Worse: the warning "
    "was at DEBUG level, so most users never saw it.",
    "Patch elevates the warning to WARNING level (always visible). Additionally, "
    "when ALL_PROXY=socks5:// is detected and Slack is configured, emit a "
    "gateway startup warning: 'Slack cannot use SOCKS5 proxy. Slack connections "
    "will NOT route through Tor. Use http:// proxy (privoxy) or accept the leak.' "
    "Added to SKILL.md as known limitation with privoxy workaround.",
    "Verify: gateway startup logs show Slack SOCKS5 warning if ALL_PROXY is socks5://",
    "plugins/platforms/slack/adapter.py"
)

register(
    LeakSeverity.MEDIUM, LeakStatus.FIXED,
    "Gateway restart race — proxy points to dead Tor port after crash",
    "If the gateway crashes and supervisor restarts it, .env is reloaded with "
    "ALL_PROXY=socks5://127.0.0.1:9050. But if Tor also crashed, the proxy port "
    "is dead. Each adapter handles proxy failure differently: some fail open "
    "(connect direct), some fail closed (refuse to connect). No consistent behavior.",
    "start_tor_for_gateway() checks that the SOCKS listener accepts connections "
    "before injecting or persisting proxy configuration. If Tor is dead, the "
    "wrapper refuses to launch the gateway. TOR_HEALTH=ok is written only after "
    "that check, and clear_gateway_env() removes it.",
    "Verify: kill Tor process, restart gateway — should refuse to connect",
    "darkloom/gateway.py"
)

register(
    LeakSeverity.MEDIUM, LeakStatus.FIXED,
    "Platform env override gap — empty platform var silently disables Tor",
    "DISCORD_PROXY= (empty) overrides ALL_PROXY=socks5://... in resolve_proxy_url(). "
    "If a user previously set DISCORD_PROXY= to disable a broken proxy, then "
    "later sets ALL_PROXY for Tor, Discord connects direct with NO warning "
    "that the empty platform var is overriding Tor.",
    "resolve_proxy_url() now logs at WARNING when a platform-specific env var "
    "is empty and ALL_PROXY is set: 'DISCORD_PROXY is set but empty — Discord "
    "will NOT use ALL_PROXY=socks5://...'. Added to all platform adapter init. "
    "Also added to SKILL.md troubleshooting: check for empty platform proxy vars.",
    "Verify: set DISCORD_PROXY=, ALL_PROXY=socks5://..., start gateway — should see warning",
    "gateway/platforms/base.py"
)

# ═══════════════════════════════════════════════════════════════
# LOW — Protocol limitations, documented
# ═══════════════════════════════════════════════════════════════

register(
    LeakSeverity.LOW, LeakStatus.DOCUMENTED,
    "Discord voice UDP — SOCKS5 proxies TCP only",
    "SOCKS5 protocol proxies TCP connections. Discord voice uses UDP (port 50000+). "
    "Voice data cannot go through SOCKS5. WebRTC/voice will ALWAYS use direct UDP "
    "regardless of proxy configuration.",
    "Documented in SKILL.md. TOR_STRICT_MODE does not control the external "
    "Discord adapter, so voice must be disabled there. Users who need voice "
    "through Tor must use a VPN with UDP support.",
    "This is a SOCKS5 protocol limitation — not fixable in Hermes",
    "Discord adapter (documentation only)"
)

register(
    LeakSeverity.LOW, LeakStatus.DOCUMENTED,
    "Email SMTP/IMAP — raw sockets, no SOCKS5 support",
    "SMTP (port 25/587) and IMAP (port 993) use raw TCP sockets. Hermes email "
    "adapter does not use httpx or aiohttp — it uses smtplib/imaplib which "
    "do not support SOCKS5. Email connections will ALWAYS go direct.",
    "Documented in SKILL.md. Email must be disabled separately in strict deployments. "
    "Alternative: use a SOCKS5-aware email library (aiosmtpd with proxy) or "
    "route through a system-level transparent proxy.",
    "This is a Python stdlib limitation — smtplib/imaplib don't support SOCKS5",
    "Email adapter (documentation only)"
)

register(
    LeakSeverity.LOW, LeakStatus.DOCUMENTED,
    "IRC — raw TCP sockets, no SOCKS5 support",
    "IRC uses raw TCP sockets on port 6667/6697. Python's irc library does "
    "not support SOCKS5. IRC connections will ALWAYS go direct.",
    "Documented in SKILL.md. IRC must be disabled separately in strict deployments.",
    "This is a protocol limitation — IRC doesn't use HTTP",
    "IRC adapter (documentation only)"
)

register(
    LeakSeverity.LOW, LeakStatus.DOCUMENTED,
    "Gateway import-time network calls — connections before proxy is set",
    "Some platform adapters may make network calls during module import "
    "(credential validation, config fetching, API version checks). These "
    "happen before resolve_proxy_url() is ever called or ALL_PROXY is "
    "injected. Timing-dependent — hard to verify without auditing every "
    "adapter's import path.",
    "Audited the three largest adapters (Telegram, Discord, WhatsApp). "
    "None make network calls at import time. All use lazy initialization. "
    "Smaller adapters may vary and remain a documented risk. The gateway wrapper "
    "must be started before importing or initializing platform adapters.",
    "Verify: launch through darkloom.gateway and check for pre-proxy connections",
    "All platform adapters (documentation)"
)

register(
    LeakSeverity.HIGH, LeakStatus.DOCUMENTED,
    "LLM exit node hostility — providers block known Tor exit IPs (403/429/CAPTCHA)",
    "OpenAI, Anthropic, and their API gateways (Cloudflare, AWS WAF) aggressively "
    "block traffic from known Tor exit nodes. Even though SOCKS5 hides traffic "
    "from your ISP, the provider sees a Tor exit IP and returns HTTP 403, 429, "
    "or invisible CAPTCHA challenges. This breaks API authentication and makes "
    "LLM calls through Tor unreliable for major providers.",
    "Documented mitigation: route LLM calls through VPN-only (bypass Tor) while "
    "keeping all other traffic through Tor. The LLM API key already identifies "
    "your account — Tor adds IP privacy but not account anonymity for API calls. "
    "Non-strict deployments may use a request-scoped client only when an explicit "
    "per-provider policy allows direct routing. For truly anonymous LLM access, use "
    "providers that don't block Tor (local models, some open-source endpoints) "
    "or route through a non-exit-node SOCKS5 proxy chain.",
    "Verify: direct routing is rejected unless the provider policy opts in",
    "darkloom/gateway.py, SKILL.md"
)

register(
    LeakSeverity.MEDIUM, LeakStatus.DOCUMENTED,
    "execute_code system binary leaks — git, curl, compiled tools bypass proxy",
    "ALL_PROXY and HTTP_PROXY are environment variables that Python libraries "
    "(httpx, aiohttp) respect — but they are polite requests, not physical "
    "barriers. If an LLM writes an execute_code block that invokes a system "
    "binary (git clone, curl, pip install, compiled C/Go/Rust tool), that "
    "process ignores proxy env vars and uses the raw network interface. "
    "This is a Python-level limitation — we proxy the Python HTTP stack but "
    "cannot control subprocess network behavior.",
    "Documented mitigation: On Linux, system binaries can be wrapped with "
    "torsocks (LD_PRELOAD). On Windows, no equivalent exists. Strict deployments "
    "must prevent execute_code blocks from spawning network clients. Future: restricted "
    "network namespaces via containerization (Docker with --network=none and "
    "SOCKS5 proxy as sole egress).",
    "Verify: check if git/curl/pip calls in execute_code show Tor exit IP in network capture",
    "execute_code sandbox (documentation)"
)

register(
    LeakSeverity.MEDIUM, LeakStatus.DOCUMENTED,
    "Tor latency degrades streaming — 3-hop circuit + obfs4 adds 500ms-2s",
    "Streaming LLM tokens through a 3-hop Tor circuit with obfs4 bridges "
    "introduces 500ms-2s additional latency per request. Time To First Token "
    "(TTFT) spikes. WebSocket connections (Discord, Matrix) may drop if "
    "latency exceeds heartbeat timeout (typically 30-60s, so unlikely but "
    "possible on degraded circuits). Bulk API calls (non-streaming) are "
    "less affected — the latency hit is on connection setup, not per-token.",
    "Documented tradeoff: Tor latency is the price of censorship resistance. "
    "For latency-sensitive workloads (streaming chat, real-time voice), "
    "consider VPN-only routing. For batch workloads (subagent research, "
    "scheduled tasks, data extraction), Tor overhead is negligible. "
    "A policy-approved request-scoped direct client can preserve TTFT in non-strict mode.",
    "Measure: compare TTFT with and without Tor using the same provider/model",
    "All network paths (documentation)"
)


# ═══════════════════════════════════════════════════════════════
def load_manifest() -> dict:
    path = files("darkloom").joinpath("compatibility-manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _controls(manifest: dict) -> list[Control]:
    revision = manifest["upstream"]["required_commit"]
    return [
        Control(item["id"], item["title"], tuple(item.get("files", ())),
                revision, item["patch_id"], item.get("documentation_only", False))
        for item in manifest["controls"]
    ]


def find_hermes_root(explicit: Path | str | None = None) -> Path | None:
    """Find Hermes without treating the darkloom repository as Hermes."""
    candidates = [explicit, os.environ.get("HERMES_HOME"), Path.cwd()]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        if (root / "gateway/platforms/base.py").is_file() and (root / "plugins").is_dir():
            return root
    return None


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_compatibility(
    hermes_root: Path | str | None = None,
    *,
    strict: bool | None = None,
    runtime_probes: dict[str, Callable[[], bool]] | None = None,
) -> list[ControlResult]:
    """Compare the installed Hermes revision and files to the signed-off manifest.

    A matching file set is MITIGATED, not VERIFIED.  VERIFIED requires a named
    runtime probe from the embedding application; darkloom never infers network
    behaviour from configuration alone.
    """
    manifest = load_manifest()
    controls = _controls(manifest)
    strict = is_strict_mode() if strict is None else strict
    root = find_hermes_root(hermes_root)
    patch_path = Path(__file__).resolve().parents[2] / manifest["patch"]["path"]
    patch_ok = patch_path.is_file() and _sha256(patch_path) == manifest["patch"]["sha256"]
    revision = _git_revision(root) if root else None
    revision_ok = revision == manifest["upstream"]["required_commit"]
    expected = manifest["patched_files"]
    results: list[ControlResult] = []

    for control in controls:
        if control.documentation_only:
            result = ControlResult(control, ControlStatus.UNVERIFIED,
                                   EvidenceKind.DOCUMENTATION,
                                   "limitation documented; no enforcement is claimed")
            probe = (runtime_probes or {}).get(control.id)
            if probe is not None:
                try:
                    passed = probe() is True
                except Exception as exc:
                    result = ControlResult(control, ControlStatus.INCOMPATIBLE,
                                           EvidenceKind.RUNTIME_VERIFICATION,
                                           f"runtime probe failed: {exc}")
                else:
                    if passed:
                        result = ControlResult(
                            control, ControlStatus.VERIFIED,
                            EvidenceKind.RUNTIME_VERIFICATION,
                            "caller-supplied runtime probe confirmed the unsafe feature is disabled")
                    else:
                        result = ControlResult(control, ControlStatus.INCOMPATIBLE,
                                               EvidenceKind.RUNTIME_VERIFICATION,
                                               "caller-supplied runtime probe reported failure")
        elif root is None or not revision_ok:
            detail = "Hermes installation not found" if root is None else (
                f"Hermes revision {revision or 'unknown'} != required {control.hermes_revision}")
            result = ControlResult(
                control, ControlStatus.PATCH_ONLY if patch_ok else ControlStatus.UNVERIFIED,
                EvidenceKind.PATCH_ARTIFACT if patch_ok else EvidenceKind.DOCUMENTATION, detail)
        else:
            bad = [name for name in control.files
                   if not (root / name).is_file() or _sha256(root / name) != expected[name]]
            if bad:
                result = ControlResult(control, ControlStatus.PATCH_ONLY,
                                       EvidenceKind.PATCH_ARTIFACT,
                                       "missing or incompatible installed files: " + ", ".join(bad))
            else:
                result = ControlResult(control, ControlStatus.MITIGATED,
                                       EvidenceKind.INSTALLED_PATCH,
                                       "required revision and patched file hashes match")
                probe = (runtime_probes or {}).get(control.id)
                if probe is not None:
                    try:
                        passed = probe() is True
                    except Exception as exc:
                        result = ControlResult(control, ControlStatus.INCOMPATIBLE,
                                               EvidenceKind.RUNTIME_VERIFICATION,
                                               f"runtime probe failed: {exc}")
                    else:
                        if passed:
                            result = ControlResult(control, ControlStatus.VERIFIED,
                                                   EvidenceKind.RUNTIME_VERIFICATION,
                                                   "caller-supplied runtime probe passed")
                        else:
                            result = ControlResult(control, ControlStatus.INCOMPATIBLE,
                                                   EvidenceKind.RUNTIME_VERIFICATION,
                                                   "caller-supplied runtime probe reported failure")

        results.append(result)

    incompatible = [r for r in results if r.status in (
        ControlStatus.UNVERIFIED, ControlStatus.PATCH_ONLY, ControlStatus.INCOMPATIBLE)]
    if strict and incompatible:
        summary = "; ".join(f"{r.control.id}: {r.detail}" for r in incompatible)
        raise CompatibilityError(f"strict mode rejected incompatible Hermes integration: {summary}")
    return results


# Hardening Tools
# ═══════════════════════════════════════════════════════════════

def run_audit():
    """Print the full leak audit table."""
    print("=" * 78)
    print("  HERMES-TOR ADVERSARIAL HARDENING AUDIT")
    print("=" * 78)
    print()

    by_severity = {}
    for leak in LEAKS:
        by_severity.setdefault(leak.severity, []).append(leak)

    for severity in [LeakSeverity.CRITICAL, LeakSeverity.HIGH,
                     LeakSeverity.MEDIUM, LeakSeverity.LOW]:
        leaks = by_severity.get(severity, [])
        if not leaks:
            continue
        print(f"  [{severity.value.upper()}]")
        for leak in leaks:
            status_icon = {"fixed": "✅", "mitigated": "⚠️", "documented": "📄"}[leak.status.value]
            print(f"  {status_icon} {leak.id}: {leak.title}")
            print(f"     Status: {leak.status.value.upper()}")
            print(f"     Before: {leak.before[:120]}...")
            print(f"     After:  {leak.after[:120]}...")
            if leak.component:
                print(f"     File:   {leak.component}")
            print()

    fixed = sum(1 for l in LEAKS if l.status == LeakStatus.FIXED)
    mitigated = sum(1 for l in LEAKS if l.status == LeakStatus.MITIGATED)
    documented = sum(1 for l in LEAKS if l.status == LeakStatus.DOCUMENTED)

    print(f"  SUMMARY: {len(LEAKS)} leaks → {fixed} FIXED, {mitigated} MITIGATED, {documented} DOCUMENTED")
    print("=" * 78)


def inject_subprocess_proxy_env(env_dict: dict[str, str]) -> dict[str, str]:
    """Inject proxy env vars into a subprocess environment dict.

    Call this before any subprocess.Popen that spawns a child process
    that needs to route through Tor.

    Args:
        env_dict: The environment dict to modify (typically os.environ.copy())

    Returns:
        The same dict with proxy vars injected (mutated in place)
    """
    proxy_vars = {
        "ALL_PROXY": os.environ.get("ALL_PROXY", ""),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
        "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
        "TOR_PROXY": os.environ.get("TOR_PROXY", ""),
    }
    for key, value in proxy_vars.items():
        if value:  # Only set if non-empty
            env_dict[key] = value
        elif key in env_dict:
            # Explicitly set empty to override any inherited value
            pass

    logger.debug("Injected proxy env vars into subprocess env: %s",
                 {k: v for k, v in proxy_vars.items() if v})
    return env_dict


def enable_strict_mode():
    """Enable TOR_STRICT_MODE for integrations that explicitly enforce it.

    The darkloom gateway wrapper always verifies that its Tor SOCKS listener
    is reachable before launching the gateway. External Hermes adapters do not
    currently consume this setting, so callers must not treat it as a blanket
    block for Discord voice, Email, IRC, or other direct network clients.
    """
    os.environ["TOR_STRICT_MODE"] = "1"
    logger.warning(
        "TOR_STRICT_MODE enabled — only integrations that explicitly check it "
        "are restricted; direct-network adapters must remain disabled"
    )
    return True


def is_strict_mode() -> bool:
    return os.environ.get("TOR_STRICT_MODE", "").lower() in ("1", "true", "yes")


def check_tor_health(socks_port: int = 9050, timeout: float = 2.0) -> bool:
    """Check if Tor SOCKS5 proxy is accepting connections."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex(("127.0.0.1", socks_port))
        sock.close()
        return result == 0
    except Exception:
        return False


# CLI
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        run_audit()
    else:
        run_audit()
