#!/usr/bin/env python3
"""Standalone BOTCOIN mining helper for the well-known skill.

This script is delivered to users who run::

    hermes skills install https://coordinator.agentmoney.net/.well-known/skills/index.json#botcoin-mining

Hermes drops it under ``~/.hermes/skills/blockchain/botcoin-mining/scripts/``
and the bundled ``SKILL.md`` calls it directly.

It is **truly standalone**: no `pip install hermes-botcoin` required for any
subcommand. The only optional dependency is ``eth-account`` — only needed
when ``BOTCOIN_SIGNER=eoa`` (i.e. you are signing locally with a private key
rather than using Bankr).

Subcommands:
    status                          Snapshot from coordinator (no auth)
    setup                           Run pre-flight checklist
    health                          Coordinator /health
    challenge [--nonce]             Fetch a challenge (writes JSON to stdout)
    submit ...                      Submit artifact + trace (broadcasts on pass)
    mine [--loop] [--solver ...]    End-to-end auto loop (auth → solve → submit → broadcast)
    claim --epochs 41,42            Claim mining rewards
    stake --amount 5000000          Stake whole BOTCOIN (approve + stake)
    unstake [--cancel]              Begin unstake / cancel pending
    withdraw                        Withdraw after cooldown

Solver providers for `mine`: venice (default, recommended), anthropic, openai,
openrouter, deepseek. Set BOTCOIN_SOLVER_PROVIDER and the matching key
(VENICE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY,
DEEPSEEK_API_KEY) in your env.

Authoritative protocol skill: https://agentmoney.net/skill.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "https://coordinator.agentmoney.net").rstrip("/")
DEFAULT_RPC = os.environ.get("BASE_RPC_URL", "https://mainnet.base.org")
TOKEN_ADDR = "0xA601877977340862Ca67f816eb079958E5bd0BA3"
MINING_V3 = "0xB2fbe0DB5A99B4E2Dd294dE64cEd82740b53A2Ea"
USER_AGENT = "botcoin-skill/1.0 (+https://coordinator.agentmoney.net)"
BASE_CHAIN_ID = 8453

SOLVER_PROVIDERS = ("venice", "anthropic", "openai", "openrouter", "deepseek")
DEFAULT_SOLVER_MODELS = {
    "venice": "zai-org-glm-5.1",
    "anthropic": "claude-opus-4-7",
    "openai": "gpt-5.1",
    "openrouter": "anthropic/claude-opus-4.7",
    "deepseek": "deepseek-reasoner",
}


# ---------------------------------------------------------------------------
# HTTP helpers


def _http(method: str, url: str, *, headers: dict | None = None, body: dict | None = None,
          timeout: int = 60) -> tuple[int, dict | str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        h["Content-Type"] = "application/json"
    if headers:
        h.update(headers)
    req = urllib.request.Request(url=url, data=data, headers=h, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                return resp.status, raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read() or b""
        try:
            return exc.code, json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return exc.code, raw.decode("utf-8", errors="replace")


def _coord_get(path: str, *, bearer: str | None = None, timeout: int = 60) -> dict:
    h = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    code, data = _http("GET", COORDINATOR_URL + path, headers=h, timeout=timeout)
    if not isinstance(data, dict):
        data = {"raw": data}
    if code >= 400:
        raise RuntimeError(f"GET {path} → {code}: {data}")
    return data


def _coord_post(path: str, body: dict, *, bearer: str | None = None, timeout: int = 60) -> dict:
    h = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    code, data = _http("POST", COORDINATOR_URL + path, headers=h, body=body, timeout=timeout)
    if not isinstance(data, dict):
        data = {"raw": data}
    if code >= 400:
        raise RuntimeError(f"POST {path} → {code}: {data}")
    return data


# ---------------------------------------------------------------------------
# Signer (EOA preferred, Bankr fallback) — same precedence as the Hermes plugin


def _signer_mode() -> str:
    forced = (os.environ.get("BOTCOIN_SIGNER") or "").strip().lower()
    if forced in ("eoa", "bankr"):
        return forced
    if forced:
        raise RuntimeError(f"unknown BOTCOIN_SIGNER mode: {forced!r}")
    if os.environ.get("BOTCOIN_MINER_KEY"):
        return "eoa"
    if os.environ.get("BANKR_API_KEY"):
        return "bankr"
    raise RuntimeError("Set BOTCOIN_MINER_KEY (preferred) or BANKR_API_KEY in your environment.")


def _eoa_address() -> str:
    from eth_account import Account
    return Account.from_key(os.environ["BOTCOIN_MINER_KEY"]).address


def _eoa_sign(message: str) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    signed = Account.sign_message(encode_defunct(text=message), private_key=os.environ["BOTCOIN_MINER_KEY"])
    sig = signed.signature.hex() if hasattr(signed.signature, "hex") else str(signed.signature)
    return sig if sig.startswith("0x") else "0x" + sig


def _bankr_get(path: str) -> dict:
    code, data = _http("GET", "https://api.bankr.bot" + path,
                       headers={"X-API-Key": os.environ["BANKR_API_KEY"]}, timeout=30)
    if not isinstance(data, dict):
        data = {"raw": data}
    if code >= 400:
        raise RuntimeError(f"bankr GET {path} → {code}: {data}")
    return data


def _bankr_post(path: str, body: dict, *, timeout: int = 60) -> dict:
    code, data = _http("POST", "https://api.bankr.bot" + path,
                       headers={"X-API-Key": os.environ["BANKR_API_KEY"]},
                       body=body, timeout=timeout)
    if not isinstance(data, dict):
        data = {"raw": data}
    if code >= 400:
        raise RuntimeError(f"bankr POST {path} → {code}: {data}")
    return data


def _bankr_address() -> str:
    if os.environ.get("BOTCOIN_MINER_ADDRESS"):
        return os.environ["BOTCOIN_MINER_ADDRESS"]
    me = _bankr_get("/agent/me")
    for w in me.get("wallets", []) or []:
        if (w.get("chain") or "").lower() == "evm":
            return w["address"]
    raise RuntimeError("Bankr: no EVM wallet on this account")


def _signer_address() -> str:
    return _eoa_address() if _signer_mode() == "eoa" else _bankr_address()


def _signer_sign(message: str) -> str:
    if _signer_mode() == "eoa":
        return _eoa_sign(message)
    return _bankr_post("/agent/sign", {"signatureType": "personal_sign", "message": message})["signature"]


def _broadcast_tx(tx: dict, *, wait: bool = True) -> dict:
    if _signer_mode() == "eoa":
        return _eoa_broadcast(tx, wait=wait)
    return _bankr_broadcast(tx, wait=wait)


def _eoa_broadcast(tx: dict, *, wait: bool) -> dict:
    """Sign + broadcast an EIP-1559 transaction via Base JSON-RPC.

    Falls back to legacy gasPrice only if EIP-1559 fees can't be inferred
    (extremely unlikely on Base, but keeps the helper robust).
    """
    from eth_account import Account
    key = os.environ["BOTCOIN_MINER_KEY"]
    sender = Account.from_key(key).address
    chain_id = int(tx.get("chainId") or BASE_CHAIN_ID)
    value = int(tx.get("value") or 0)

    def rpc(method, params):
        code, data = _http(
            "POST", DEFAULT_RPC,
            body={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        if code >= 400 or "error" in (data if isinstance(data, dict) else {}):
            raise RuntimeError(f"rpc {method}: {data}")
        return data["result"]

    nonce = int(rpc("eth_getTransactionCount", [sender, "pending"]), 16)
    gas_estimate = int(rpc(
        "eth_estimateGas",
        [{"from": sender, "to": tx["to"], "data": tx["data"], "value": hex(value)}],
    ), 16)
    gas_limit = int(gas_estimate * 1.2)

    # EIP-1559 fee suggestion
    try:
        priority_hex = rpc("eth_maxPriorityFeePerGas", [])
        priority = int(priority_hex, 16)
        if priority < 1_000_000_000:
            priority = 1_000_000_000
        block = rpc("eth_getBlockByNumber", ["latest", False])
        base_fee = int(block.get("baseFeePerGas") or "0x0", 16)
        max_fee = 2 * base_fee + priority
        unsigned = {
            "type": 2, "to": tx["to"], "value": value, "gas": gas_limit,
            "maxFeePerGas": max_fee, "maxPriorityFeePerGas": priority,
            "nonce": nonce, "data": tx["data"], "chainId": chain_id,
        }
        signed = Account.sign_transaction(unsigned, private_key=key)
    except Exception:
        gas_price = int(rpc("eth_gasPrice", []), 16)
        unsigned = {
            "to": tx["to"], "value": value, "gas": gas_limit, "gasPrice": gas_price,
            "nonce": nonce, "data": tx["data"], "chainId": chain_id,
        }
        signed = Account.sign_transaction(unsigned, private_key=key)

    raw = signed.raw_transaction.hex() if hasattr(signed, "raw_transaction") else signed.rawTransaction.hex()
    if not raw.startswith("0x"):
        raw = "0x" + raw
    tx_hash = rpc("eth_sendRawTransaction", [raw])
    out = {"transactionHash": tx_hash, "status": "pending", "from": sender}
    if not wait:
        return out
    deadline = time.time() + 180
    while time.time() < deadline:
        receipt = rpc("eth_getTransactionReceipt", [tx_hash])
        if receipt:
            out["status"] = "success" if receipt.get("status") == "0x1" else "reverted"
            out["blockNumber"] = receipt.get("blockNumber")
            out["gasUsed"] = receipt.get("gasUsed")
            return out
        time.sleep(2)
    out["status"] = "timeout"
    return out


def _bankr_broadcast(tx: dict, *, wait: bool) -> dict:
    body = {
        "transaction": {
            "to": tx["to"], "chainId": int(tx.get("chainId") or BASE_CHAIN_ID),
            "value": str(tx.get("value") or "0"), "data": tx["data"],
        },
        "description": "BOTCOIN mining transaction",
        "waitForConfirmation": bool(wait),
    }
    resp = _bankr_post("/agent/submit", body, timeout=180)
    return {"transactionHash": resp.get("transactionHash") or resp.get("hash"),
            "status": resp.get("status", "submitted"), "raw": resp}


# ---------------------------------------------------------------------------
# Auth handshake


def _extract_addr(message: str) -> str | None:
    for line in message.splitlines():
        if line.startswith("Address:"):
            v = line.split(":", 1)[1].strip()
            if v.startswith("0x") and len(v) == 42:
                return v
    return None


def _auth() -> tuple[str, str]:
    miner = _signer_address()
    nonce = _coord_post("/v1/auth/nonce", {"miner": miner})
    canonical = _extract_addr(nonce["message"]) or miner
    sig = _signer_sign(nonce["message"])
    if not sig.startswith("0x"):
        sig = "0x" + sig
    verify = _coord_post(
        "/v1/auth/verify",
        {"miner": canonical, "message": nonce["message"], "signature": sig},
    )
    return miner, verify["token"]


# ---------------------------------------------------------------------------
# Trace + answer normalization (mirrors the canonical v3 schema in
# https://agentmoney.net/skill.md — string step_id, no `task`/`step` fields)


def _normalize_trace(trace) -> list[dict]:
    if not trace or not isinstance(trace, list):
        return []
    out: list[dict] = []
    used: set[str] = set()
    for i, row in enumerate(trace):
        if not isinstance(row, dict):
            continue
        action = str(row.get("action", "extract_fact"))
        sid_raw = row.get("step_id")
        if isinstance(sid_raw, str) and sid_raw.strip():
            sid = sid_raw.strip()
        elif isinstance(sid_raw, (int, float)):
            sid = f"s{int(sid_raw)}"
        else:
            prefix = {"extract_fact": "e", "compute_logic": "c",
                      "revision": "rev", "backtrack": "back",
                      "note": "n", "verify": "v"}.get(action, "s")
            sid = f"{prefix}{i + 1}"
        while sid in used:
            sid = f"{sid}_"
        used.add(sid)
        item: dict = {"step_id": sid, "action": action}
        if action == "extract_fact":
            item["targetEntity"] = str(row.get("targetEntity", "") or "")
            item["attribute"] = str(row.get("attribute", "") or "")
            item["valueExtracted"] = row.get("valueExtracted")
            item["source"] = str(row.get("source", "") or "")
        elif action == "compute_logic":
            op = str(row.get("operation", "") or "")
            if op == "roundNearest":
                op = "round_nearest"
            item["operation"] = op
            item["inputs"] = list(row.get("inputs") or [])
            item["result"] = row.get("result")
        else:
            for k, v in row.items():
                if k in ("action", "step", "task", "step_id"):
                    continue
                item[k] = v
        out.append(item)
    return out


def _normalize_answers(answers) -> dict | None:
    if not answers:
        return None
    if isinstance(answers, dict):
        return {str(k): "" if v is None else str(v) for k, v in answers.items() if k}
    if isinstance(answers, (list, tuple)):
        return {f"q{i + 1:02d}": "" if v is None else str(v) for i, v in enumerate(answers)}
    return None


# ---------------------------------------------------------------------------
# Vendored solver — every supported provider, no plugin dep


SYSTEM_PROMPT = (
    "You are an autonomous BOTCOIN miner solving a proof-of-inference challenge. "
    "Read the document, answer every question, derive every constraint, and produce "
    "a single-line artifact that satisfies ALL constraints simultaneously. Then build "
    "a structured v3 reasoning trace (extract_fact + compute_logic steps with string "
    'step_id values like "e1", "c1") that cites paragraph_N references. '
    "Respond with a single JSON object and nothing else: "
    '{"artifact": "<single-line>", "reasoningTrace": [<steps>], '
    '"submittedAnswers": {"q01": "...", "q05": "...", ...}}.'
)


def _build_solver_prompt(challenge: dict) -> str:
    doc = challenge.get("doc", "")
    questions = challenge.get("questions") or []
    constraints = challenge.get("constraints") or []
    entities = challenge.get("entities") or challenge.get("companies") or []
    instructions = challenge.get("solveInstructions", "")
    trace_ref = challenge.get("traceReference") or challenge.get("documentMap")

    def _norm(x):
        if isinstance(x, dict):
            return str(x.get("text") or x.get("key") or x)
        return str(x)

    q_lines = "\n".join(f"Q{i + 1}: {_norm(q)}" for i, q in enumerate(questions))
    c_lines = "\n".join(f"C{i + 1}: {_norm(c)}" for i, c in enumerate(constraints))
    trace_block = (
        "\n\n## TraceReference (use these paragraphs for citations)\n" + json.dumps(trace_ref, indent=2)
        if trace_ref else ""
    )
    return (
        "You are solving a BOTCOIN mining challenge. Produce ONE single-line artifact "
        "that satisfies every listed constraint, plus a v3 reasoning trace.\n\n"
        f"## Document (paragraphs are pre-numbered as paragraph_N — cite those exact labels)\n{doc}\n\n"
        f"## Questions\n{q_lines}\n\n"
        f"## Valid entity names\n{json.dumps(entities)}{trace_block}\n\n"
        f"## Constraints\n{c_lines}\n\n"
        f"## Solve instructions\n{instructions}\n\n"
        '## Output format — exactly one JSON object\n'
        '{"artifact": "<single-line>", "reasoningTrace": [<steps>], "submittedAnswers": {"q01": "..."}}\n\n'
        "## Trace step shape (the coordinator validates this strictly)\n"
        '- extract_fact: {"step_id":"e1","action":"extract_fact",'
        '"targetEntity":"<entity>","attribute":"<canonical-attr>","valueExtracted":<value>,'
        '"source":"paragraph_N"}\n'
        '- compute_logic: {"step_id":"c1","action":"compute_logic",'
        '"operation":"add|sum|subtract|multiply|divide|mod|max|min|average|next_prime|round|round_nearest|abs_diff|ratio|count|compare_equal|compare_greater_than|compare_less_than",'
        '"inputs":[<prior step_id strings or literal numbers>],"result":<value>}\n'
        "Step IDs are STRINGS, not integers. inputs reference PRIOR step_ids by string."
    )


def _llm_call(url: str, headers: dict, body: dict, *, timeout: int = 600) -> dict:
    """Single-attempt LLM call with structured 4xx/5xx error surfacing."""
    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_preview = ""
        try:
            body_preview = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"LLM HTTP {exc.code}: {body_preview}") from exc


def _solve_venice(prompt: str, model: str, max_tokens: int) -> str:
    key = os.environ.get("VENICE_API_KEY")
    if not key:
        raise RuntimeError("VENICE_API_KEY required for venice solver — https://venice.ai/settings/api")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "venice_parameters": {
            "include_venice_system_prompt": False,
            "enable_web_search": "off",
        },
    }
    data = _llm_call(
        "https://api.venice.ai/api/v1/chat/completions",
        {"Authorization": f"Bearer {key}"},
        body,
    )
    msg = (data.get("choices") or [{}])[0].get("message", {})
    return msg.get("content", "") or msg.get("reasoning_content", "")


def _solve_anthropic(prompt: str, model: str, max_tokens: int) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY required for anthropic solver")
    is_47 = "opus-4-7" in model or "claude-4-7" in model or model.startswith("claude-opus-4-7")
    body: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_47:
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": "high"}
    else:
        body["thinking"] = {"type": "enabled", "budget_tokens": 12000}
    data = _llm_call(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
        body,
    )
    out = ""
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            out += block.get("text", "")
    if not out:
        raise RuntimeError(
            f"anthropic returned no text — stop_reason={data.get('stop_reason')} "
            f"content_types={[b.get('type') for b in (data.get('content') or [])]}"
        )
    return out


def _solve_oai_compat(prompt: str, model: str, max_tokens: int, *, base_url: str, env_key: str) -> str:
    key = os.environ.get(env_key)
    if not key:
        raise RuntimeError(f"{env_key} required for this solver")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = _llm_call(
        f"{base_url.rstrip('/')}/chat/completions",
        {"Authorization": f"Bearer {key}"},
        body,
    )
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def _solve(challenge: dict, provider: str, model: str | None, max_tokens: int) -> tuple[str, list, Any, str]:
    prompt = _build_solver_prompt(challenge)
    provider = (provider or "venice").lower()
    if provider not in SOLVER_PROVIDERS:
        raise RuntimeError(f"unsupported solver provider: {provider!r}")
    model = model or DEFAULT_SOLVER_MODELS[provider]

    if provider == "venice":
        text = _solve_venice(prompt, model, max_tokens)
    elif provider == "anthropic":
        text = _solve_anthropic(prompt, model, max_tokens)
    elif provider == "openai":
        text = _solve_oai_compat(prompt, model, max_tokens,
                                 base_url="https://api.openai.com/v1", env_key="OPENAI_API_KEY")
    elif provider == "openrouter":
        text = _solve_oai_compat(prompt, model, max_tokens,
                                 base_url="https://openrouter.ai/api/v1", env_key="OPENROUTER_API_KEY")
    else:  # deepseek
        text = _solve_oai_compat(prompt, model, max_tokens,
                                 base_url="https://api.deepseek.com/v1", env_key="DEEPSEEK_API_KEY")

    artifact, trace, answers = _parse_solver_output(text)
    return artifact, trace, answers, f"{provider}/{model}"


def _parse_solver_output(text: str) -> tuple[str, list, Any]:
    """Robust JSON extraction: raw_decode brace scan + code-fence stripping."""
    if not text:
        raise RuntimeError("solver returned no text")
    stripped = text.strip()
    if stripped.startswith("```"):
        without_open = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if without_open.rstrip().endswith("```"):
            without_open = without_open.rstrip()[:-3]
        stripped = without_open.strip()

    candidates: list[dict] = []
    try:
        c = json.loads(stripped)
        if isinstance(c, dict):
            candidates.append(c)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
        i = end

    for parsed in candidates:
        if not isinstance(parsed.get("artifact"), str):
            continue
        artifact = parsed["artifact"].strip()
        trace = parsed.get("reasoningTrace") or parsed.get("reasoningLog") or []
        answers = parsed.get("submittedAnswers")
        if artifact and isinstance(trace, list):
            return artifact, trace, answers

    m = re.search(r"\"artifact\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"", text)
    if m:
        return m.group(1).encode("utf-8").decode("unicode_escape").strip(), [], None
    raise RuntimeError(f"could not parse artifact: {text[:600]}")


# ---------------------------------------------------------------------------
# Subcommand bodies


def cmd_status(args):
    print(json.dumps({
        "epoch": _coord_get("/v1/epoch"),
        "stats": _coord_get("/v1/stats"),
        "totalStaked": _coord_get("/v1/frontend/total-staked"),
    }, indent=2))


def cmd_health(args):
    print(json.dumps(_coord_get("/health"), indent=2))


def cmd_setup(args):
    issues = []
    info: dict[str, Any] = {}
    try:
        info["signer_mode"] = _signer_mode()
        info["miner"] = _signer_address()
    except Exception as exc:
        issues.append(f"signer: {exc}")
        info["signer_mode"] = None
    try:
        info["epoch"] = _coord_get("/v1/epoch")
        info["coordinator_reachable"] = True
    except Exception as exc:
        issues.append(f"coordinator: {exc}")
        info["coordinator_reachable"] = False
    print(json.dumps({"info": info, "issues": issues}, indent=2))
    if issues:
        sys.exit(1)


def cmd_challenge(args):
    miner, bearer = _auth()
    nonce = (args.nonce or uuid.uuid4().hex[:32])
    ch = _coord_get(f"/v1/challenge?miner={miner}&nonce={nonce}", bearer=bearer)
    print(json.dumps({"miner": miner, "nonce_echoed": nonce, "challenge": ch}, indent=2))


def cmd_submit(args):
    artifact = open(args.artifact_file).read().strip() if args.artifact_file else (args.artifact or "")
    trace = json.loads(open(args.trace_file).read()) if args.trace_file else json.loads(args.trace or "[]")
    if not artifact:
        print("ERROR: provide --artifact or --artifact-file", file=sys.stderr)
        sys.exit(2)
    miner, bearer = _auth()
    body = {
        "miner": miner,
        "challengeId": args.challenge_id,
        "nonce": args.nonce,
        "challengeManifestHash": args.manifest_hash,
        "artifact": artifact,
        "reasoningTrace": _normalize_trace(trace),
        "modelVersion": args.model_version,
    }
    if args.submitted_answers:
        body["submittedAnswers"] = _normalize_answers(json.loads(args.submitted_answers))
    result = _coord_post("/v1/submit", body, bearer=bearer, timeout=180)
    out: dict[str, Any] = {"submit": result}
    if result.get("pass"):
        if result.get("transaction"):
            out["receipt"] = _broadcast_tx(result["transaction"], wait=True)
        if result.get("vouchTransaction"):
            try:
                out["vouch"] = _broadcast_tx(result["vouchTransaction"], wait=False)
            except Exception as exc:  # vouch is fire-and-forget — never fail the submit
                out["vouch"] = {"ok": False, "error": str(exc)}
    print(json.dumps(out, indent=2))


def cmd_claim(args):
    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    cd = _coord_get(f"/v1/claim-calldata?epochs={','.join(str(e) for e in epochs)}")
    out = {"claim": _broadcast_tx(cd["transaction"], wait=True)}
    if not args.no_bonus:
        try:
            statuses = _coord_get(f"/v1/bonus/status?epochs={','.join(str(e) for e in epochs)}")
            statuses = statuses if isinstance(statuses, list) else [statuses]
            bonus = [int(s["epochId"]) for s in statuses
                     if isinstance(s, dict) and s.get("isBonusEpoch") and s.get("claimsOpen")]
        except Exception:
            bonus = []
        if bonus:
            bcd = _coord_get(f"/v1/bonus/claim-calldata?epochs={','.join(str(e) for e in bonus)}")
            out["bonus"] = _broadcast_tx(bcd["transaction"], wait=True)
    print(json.dumps(out, indent=2))


def cmd_stake(args):
    whole = int(str(args.amount).replace(",", "").strip())
    wei = whole * (10 ** 18)
    approve = _coord_get(f"/v1/stake-approve-calldata?amount={wei}")
    stake = _coord_get(f"/v1/stake-calldata?amount={wei}")
    print(json.dumps({
        "approve": _broadcast_tx(approve["transaction"], wait=True),
        "stake": _broadcast_tx(stake["transaction"], wait=True),
    }, indent=2))


def cmd_unstake(args):
    if args.cancel:
        print("Call cancelUnstake() on MiningContractV3 directly: " + MINING_V3, file=sys.stderr)
        sys.exit(2)
    cd = _coord_get("/v1/unstake-calldata")
    print(json.dumps(_broadcast_tx(cd["transaction"], wait=True), indent=2))


def cmd_withdraw(args):
    cd = _coord_get("/v1/withdraw-calldata")
    print(json.dumps(_broadcast_tx(cd["transaction"], wait=True), indent=2))


def cmd_mine(args):
    """End-to-end mining: auth → challenge → solve → submit → broadcast (multi-pass)."""
    attempts = 0
    while True:
        attempts += 1
        out = _mine_once(
            provider=args.solver, model=args.model, max_tokens=args.max_tokens,
            multipass_max=args.multipass_max,
        )
        print(json.dumps(out, default=str, indent=2))
        if not args.loop:
            return
        if args.max_attempts and attempts >= args.max_attempts:
            return
        time.sleep(max(60, args.cooldown))


def _mine_once(*, provider: str, model: str | None, max_tokens: int, multipass_max: int) -> dict:
    miner, bearer = _auth()
    nonce = uuid.uuid4().hex[:32]
    ch = _coord_get(f"/v1/challenge?miner={miner}&nonce={nonce}", bearer=bearer)
    cid = ch.get("challengeId")
    manifest = ch.get("challengeManifestHash", "")

    last_feedback = None
    submit_result = None
    for pass_idx in range(multipass_max):
        ch_for_solver = ch
        if pass_idx > 0 and last_feedback:
            hint = (
                f"\n\n# Prior attempt feedback (retry {pass_idx + 1}/{multipass_max})\n"
                f"constraintsPassed={last_feedback.get('constraintsPassed')}/"
                f"{last_feedback.get('constraintsTotal')}; "
                f"questionAnswersCorrect={last_feedback.get('questionAnswersCorrect')}/"
                f"{last_feedback.get('questionAnswersTotal')}. "
                "Re-derive every constraint from scratch. Add a step_id 'rev1' revision step.\n"
            )
            ch_for_solver = {**ch, "solveInstructions": (ch.get("solveInstructions") or "") + hint}
        try:
            artifact, raw_trace, raw_answers, model_ver = _solve(
                ch_for_solver, provider, model, max_tokens
            )
        except Exception as exc:
            return {"ok": False, "stage": "solve", "miner": miner, "challenge_id": cid,
                    "pass_idx": pass_idx + 1, "error": str(exc)}

        body = {
            "miner": miner, "challengeId": cid, "nonce": nonce,
            "challengeManifestHash": manifest, "artifact": artifact,
            "reasoningTrace": _normalize_trace(raw_trace),
            "modelVersion": model_ver,
        }
        ans = _normalize_answers(raw_answers)
        if ans:
            body["submittedAnswers"] = ans
        submit_result = _coord_post("/v1/submit", body, bearer=bearer, timeout=180)
        if submit_result.get("pass"):
            break
        last_feedback = submit_result
        if not submit_result.get("retryAllowed"):
            break

    if not submit_result or not submit_result.get("pass"):
        return {"ok": False, "stage": "verification", "miner": miner,
                "challenge_id": cid, "result": submit_result}

    out: dict[str, Any] = {
        "ok": True, "stage": "complete", "miner": miner, "challenge_id": cid,
        "domain": ch.get("challengeDomain"), "epoch_id": submit_result.get("receipt", {}).get("epochId"),
        "credits_earned": ch.get("creditsPerSolve"),
    }
    if submit_result.get("transaction"):
        out["receipt"] = _broadcast_tx(submit_result["transaction"], wait=True)
    if submit_result.get("vouchTransaction"):
        try:
            out["vouch"] = _broadcast_tx(submit_result["vouchTransaction"], wait=False)
        except Exception as exc:
            out["vouch"] = {"ok": False, "error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# Argument parser


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="botcoin_client", description="BOTCOIN mining helper.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("health").set_defaults(func=cmd_health)
    sub.add_parser("setup").set_defaults(func=cmd_setup)

    p_ch = sub.add_parser("challenge")
    p_ch.add_argument("--nonce")
    p_ch.set_defaults(func=cmd_challenge)

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--challenge-id", required=True)
    p_sub.add_argument("--nonce", required=True)
    p_sub.add_argument("--manifest-hash", required=True)
    p_sub.add_argument("--artifact")
    p_sub.add_argument("--artifact-file")
    p_sub.add_argument("--trace")
    p_sub.add_argument("--trace-file")
    p_sub.add_argument("--model-version", required=True)
    p_sub.add_argument("--submitted-answers", help="JSON object or array")
    p_sub.set_defaults(func=cmd_submit)

    p_claim = sub.add_parser("claim")
    p_claim.add_argument("--epochs", required=True, help="Comma-separated, e.g. 41,42")
    p_claim.add_argument("--no-bonus", action="store_true")
    p_claim.set_defaults(func=cmd_claim)

    p_stake = sub.add_parser("stake")
    p_stake.add_argument("--amount", required=True, help="Whole BOTCOIN, e.g. 5000000")
    p_stake.set_defaults(func=cmd_stake)

    p_un = sub.add_parser("unstake")
    p_un.add_argument("--cancel", action="store_true")
    p_un.set_defaults(func=cmd_unstake)

    p_wd = sub.add_parser("withdraw")
    p_wd.set_defaults(func=cmd_withdraw)

    p_mine = sub.add_parser("mine", help="End-to-end mining loop (auth → solve → submit → broadcast)")
    p_mine.add_argument("--loop", action="store_true")
    p_mine.add_argument("--max-attempts", type=int, default=0)
    p_mine.add_argument("--cooldown", type=int, default=65)
    p_mine.add_argument("--solver",
                        default=os.environ.get("BOTCOIN_SOLVER_PROVIDER", "venice"),
                        choices=list(SOLVER_PROVIDERS),
                        help="LLM provider (default: venice — privacy-by-default, OpenAI-compatible)")
    p_mine.add_argument("--model", default=os.environ.get("BOTCOIN_SOLVER_MODEL"),
                        help="Override the per-provider default model")
    p_mine.add_argument("--max-tokens", type=int, default=32000)
    p_mine.add_argument("--multipass-max", type=int, default=3)
    p_mine.set_defaults(func=cmd_mine)

    return ap


def main() -> int:
    args = _build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
