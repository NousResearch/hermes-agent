"""Request bodies for ``PATCH /v1/secrets/self`` and the real secrets plane's verdict on each.

:func:`cases` builds the bodies (exact wire bytes, deterministic). The verdicts in
``plane_patch_verdicts.json`` are gateway-gateway's own: running this file with a gateway-gateway
checkout (``node_modules`` installed) feeds every body through its router's order of checks (body
limit, ``JSON.parse``, ``parseProfilePatch`` from ``src/secretsPlane/patch.ts``, the level's entry
count) and rewrites the fixture, recording the commit:

    python -m tests.plugins.secret_store.plane_patch_cases ~/src/gateway-gateway

``test_remote_plane_parity.py`` checks :func:`plugins.secret_store.remote.limits.validate_patch`
against the fixture (and against the checkout itself when ``HERMES_TEST_GATEWAY_GATEWAY`` names one).
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).with_name("plane_patch_verdicts.json")
RESERVED = ROOT / "plugins/secret_store/remote/vendor/conformance/secrets-reserved-names.json"
KIB64 = 64 * 1024
MIB = 1024 * 1024

# The router's order of checks for PATCH /v1/secrets/self around parseProfilePatch (router.ts):
# bodyLimit (secret_body_too_large), readJson (secret_request_invalid), then the level's entries.
HARNESS = """
import {parseProfilePatch} from %(patch)s;
import {MAX_BODY_BYTES, MAX_ENTRIES_PER_LEVEL} from %(types)s;
let input = ''; for await (const c of process.stdin) input += c;
const out = [];
for (const line of input.trim().split('\\n')) {
  const bytes = Buffer.from(line, 'base64');
  try {
    if (bytes.length > MAX_BODY_BYTES) throw {code: 'secret_body_too_large'};
    let raw;
    try { raw = JSON.parse(new TextDecoder().decode(bytes)); } catch { throw {code: 'secret_request_invalid'}; }
    const p = parseProfilePatch(raw);
    const kept = p.set.size + [...p.auth.values()].filter((c) => c.record !== null).length;
    if (kept > MAX_ENTRIES_PER_LEVEL) throw {code: 'secret_level_too_large'};
    out.push('ok');
  } catch (e) { out.push(e.code || String(e)); }
}
console.log(JSON.stringify(out));
"""


def _body(env=None, auth=None, **extra) -> bytes:
    body = {"expectedProfileVersion": 0, "set": env or {},
            "auth": {r: {"expectedVersion": 0, "record": rec} for r, rec in (auth or {}).items()}, **extra}
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _record_of(size: int) -> dict:
    return {"t": "y" * (size - len('{"t":""}'))}


def _raw_record(number: str) -> bytes:
    return b'{"expectedProfileVersion":0,"auth":{"openai-codex":{"expectedVersion":0,"record":{"n":%s}}}}' % \
        number.encode()


def cases() -> list[tuple[str, bytes]]:
    out = [("ok", _body({"RV_A": "v"})), ("ok_auth_only_no_version", b'{"auth":{"codex":{"expectedVersion":0,'
                                                                    b'"record":{"t":"v"}}}}')]
    for name in ["1A", "A B", "A" * 129, "A-B", "\u00c9", ""]:
        out.append((f"env_invalid_{name[:12]!r}", _body({name: "v"})))
    out.append(("env_name_128", _body({"A" * 128: "v"})))
    for rid in ["My", "-x", "a" * 65, "a b", "_a"]:
        out.append((f"auth_invalid_{rid[:12]!r}", _body(auth={rid: {"t": "v"}})))
    out.append(("auth_id_64", _body(auth={"a" * 64: {"t": "v"}})))
    names = json.loads(RESERVED.read_text(encoding="utf-8-sig"))
    exact = names["globalEnvExact"] + names["bootstrapEnvExact"] + names["bootstrapContract"]["envExact"]
    for name in exact:
        out.append((f"reserved_{name}", _body({name: "v"})))
    for prefix in names["globalEnvPrefixes"] + names["bootstrapEnvPrefixes"]:
        out.append((f"reserved_prefix_{prefix}", _body({prefix + "X": "v"})))
        out.append((f"near_prefix_{prefix}", _body({prefix.lower() + "X": "v"})))
    for name in names["launchOnlyAnyCase"]:
        out.append((f"reserved_any_case_{name}", _body({name.lower(): "v"})))
        out.append((f"reserved_mixed_case_{name}", _body({name.title(): "v"})))
    for rid in names["bootstrapContract"]["authProviders"]:
        out.append((f"reserved_auth_{rid}", _body(auth={rid: {"t": "v"}})))
    for name in ["path", "PATHX", "PATH_", "hermes_profile", "Home", "HERMES_HOMES"]:
        out.append((f"near_reserved_{name}", _body({name: "v"})))
    out.append(("reserved_in_unset", _body(unset=["PATH"])))
    for size in (KIB64 - 1, KIB64, KIB64 + 1):
        out.append((f"env_ascii_{size}", _body({"RV_X": "x" * size})))
        out.append((f"auth_record_{size}", _body(auth={"codex": _record_of(size)})))
    out.append(("env_two_byte_64k", _body({"RV_X": "\u00e9" * (KIB64 // 2)})))
    out.append(("env_two_byte_over", _body({"RV_X": "\u00e9" * (KIB64 // 2) + "x"})))
    pad = KIB64 - len('{"n":1e+21,"t":""}')            # JSON.stringify writes 1e21 as 1e+21
    out.append(("auth_record_numbers_at_limit", _body(auth={"codex": {"n": 1e21, "t": "y" * pad}})))
    out.append(("auth_record_numbers_over", _body(auth={"codex": {"n": 1e21, "t": "y" * (pad + 1)}})))
    for label, number in [("overflow", "1e400"), ("negative_overflow", "-1e400"), ("int_beyond_double", "1" + "0" * 400),
                          ("nan", "NaN"), ("infinity", "Infinity"), ("neg_infinity", "-Infinity"),
                          ("max_double", "1.7976931348623157e308"), ("int_2_53_plus_1", "9007199254740993"),
                          ("tiny", "5e-324"), ("underflow", "1e-400")]:
        out.append((f"number_{label}", _raw_record(number)))
    out.append(("value_not_utf8", b'{"expectedProfileVersion":0,"set":{"RV_X":"\xff"}}'))
    out.append(("value_lone_surrogate_escape", b'{"expectedProfileVersion":0,"set":{"RV_X":"\\ud800"}}'))
    out.append(("record_lone_surrogate_escape", _raw_record('"\\udc00"')))
    out.append(("value_not_string", b'{"expectedProfileVersion":0,"set":{"RV_X":1}}'))
    out.append(("record_list", _body(auth={"codex": []})))
    out.append(("record_string", _body(auth={"codex": "s"})))
    out.append(("record_null", _body(auth={"codex": None})))
    out.append(("change_shape", b'{"auth":{"codex":{"record":{}}}}'))
    out.append(("change_version_negative", b'{"auth":{"codex":{"expectedVersion":-1,"record":{}}}}'))
    out.append(("change_version_float", b'{"auth":{"codex":{"expectedVersion":1.5,"record":{}}}}'))
    out.append(("change_extra_key", b'{"auth":{"codex":{"expectedVersion":0,"record":{},"x":1}}}'))
    out.append(("set_without_version", b'{"set":{"RV_A":"v"}}'))
    out.append(("version_too_big", b'{"expectedProfileVersion":2147483648,"set":{"RV_A":"v"}}'))
    out.append(("version_max", b'{"expectedProfileVersion":2147483647,"set":{"RV_A":"v"}}'))
    out.append(("unknown_key", b'{"expectedProfileVersion":0,"set":{"RV_A":"v"},"x":1}'))
    out.append(("set_list", b'{"expectedProfileVersion":0,"set":[]}'))
    out.append(("nothing_to_write", _body()))
    out.append(("set_and_unset", _body({"RV_A": "v"}, unset=["RV_A"])))
    # Versions are integers by Number.isInteger: an integral JSON number of any spelling counts.
    for label, number in [("0.0", "0.0"), ("-0.0", "-0.0"), ("1.0", "1.0"), ("max_float", "2147483647.0"),
                          ("1e0", "1e0"), ("2.0e1", "2.0e1"), ("-0", "-0"), ("1e-400", "1e-400"),
                          ("max_rounds_down", "2147483647.0000001"), ("max_plus_float", "2147483648.0"),
                          ("0.5", "0.5"), ("true", "true"), ("null", "null"), ("string", '"0"'), ("list", "[0]"),
                          ("1e400", "1e400"), ("-1.0", "-1.0")]:
        out.append((f"version_{label}", b'{"expectedProfileVersion":%s,"set":{"RV_X":"v"}}' % number.encode()))
        out.append((f"auth_version_{label}", b'{"auth":{"codex":{"expectedVersion":%s,"record":{}}}}' %
                    number.encode()))
    # `unset` holds names: any other JSON value in it is refused, never raised on.
    for label, entry in [("object", "{}"), ("list", "[]"), ("null", "null"), ("number", "1"), ("true", "true"),
                         ("nested", "[[{}]]")]:
        out.append((f"unset_{label}", b'{"expectedProfileVersion":0,"set":{"RV_X":"v"},"unset":[%s]}' %
                    entry.encode()))
    out.append(("unset_deep", b'{"expectedProfileVersion":0,"set":{"RV_X":"v"},"unset":%s%s}' %
                (b"[" * 100_000, b"]" * 100_000)))
    # The plane reads the bytes as Request.text() and JSON.parse do: a BOM dropped, any depth.
    out.append(("bom", b"\xef\xbb\xbf" + _body({"RV_A": "v"})))
    out.append(("two_boms", b"\xef\xbb\xbf\xef\xbb\xbf" + _body({"RV_A": "v"})))
    for depth in (999, 1000, 5000, 32_000):
        out.append((f"record_depth_{depth}", _raw_record("[" * depth + "]" * depth)))
    out.append(("record_depth_over_64k", _raw_record("[" * 32_800 + "]" * 32_800)))
    out.append(("record_deep_objects", _raw_record('{"a":' * 5000 + "0" + "}" * 5000)))
    out.append(("value_5000_digit_number", b'{"expectedProfileVersion":0,"set":{"RV_X":%s}}' % (b"1" * 5000)))
    out.append(("record_5000_digit_number", _raw_record("1" * 5000)))
    out.append(("not_object", b"[]"))
    out.append(("not_json", b"{"))
    for n in (999, 1000, 1001):
        out.append((f"count_env_{n}", _body({f"RV_{i}": "v" for i in range(n)})))
    out.append(("count_mixed_1001", _body({f"RV_{i}": "v" for i in range(500)},
                                          {f"p{i}": {"t": "v"} for i in range(501)})))
    out.append(("count_null_records", _body({f"RV_{i}": "v" for i in range(1000)}, {"p0": None})))
    for label, n in [("under", 15), ("over", 17)]:
        out.append((f"body_{label}_1mib", _body({f"RV_V{i}": "z" * (KIB64 - 8) for i in range(n)})))
    for label, extra in [("exactly_1mib", 0), ("1mib_plus_1", 1)]:
        names = [f"RV_V{i:02}" for i in range(16)]
        pad = MIB - len(_body(dict.fromkeys(names, ""))) + extra
        out.append((f"body_{label}", _body({n: "z" * (pad // 16 + (pad % 16 if i == 0 else 0))
                                            for i, n in enumerate(names)})))
    return out


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plane_verdicts(checkout: Path, bodies: list[bytes]) -> list[str]:
    """gateway-gateway's verdict on each body ("ok" or its error code), by its own code."""
    src = Path(checkout) / "src" / "secretsPlane"
    code = HARNESS % {"patch": json.dumps(str(src / "patch.ts")), "types": json.dumps(str(src / "types.ts"))}
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "plane-parity.mjs"
        harness.write_text(code, encoding="utf-8")
        run = subprocess.run(["node", "--import", "tsx", str(harness)], cwd=checkout, check=True,
                             capture_output=True, text=True,
                             input="\n".join(base64.b64encode(b).decode() for b in bodies))
    return json.loads(run.stdout)


def main(checkout: str) -> None:
    built = cases()
    verdicts = plane_verdicts(Path(checkout), [b for _id, b in built])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True,
                            text=True).stdout.strip()
    FIXTURE.write_text(json.dumps({
        "source": {"repo": "NousResearch/gateway-gateway", "commit": commit,
                   "files": ["src/secretsPlane/patch.ts", "src/secretsPlane/reservedNames.ts",
                             "src/secretsPlane/types.ts", "src/secretsPlane/router.ts (order of checks)"],
                   "generator": "python -m tests.plugins.secret_store.plane_patch_cases <checkout>"},
        "cases": {cid: {"sha256": digest(b), "plane": v} for (cid, b), v in zip(built, verdicts)}},
        indent=1, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
