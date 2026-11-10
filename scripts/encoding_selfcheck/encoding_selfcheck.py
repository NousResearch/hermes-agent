#!/usr/bin/env python3
"""编码契约自检套件 —— 独立可分发（**零 import 本仓**）。

面向**在本仓之外**运行的调用方 runtime（agent runtime / IDE 面板 / 外部集成）。
本文件可以整份复制进任意仓外 runtime 直接运行, 不依赖 AgentSwarm。

三个模式:

    python3 encoding_selfcheck.py probe                        # 本机环境实测
    python3 encoding_selfcheck.py static  <root> [options]     # 自身读改写点扫描
    python3 encoding_selfcheck.py mojibake <path> [<path>...]  # 存量乱码审计

契约 (给调用方的 8 条硬规则): specs/payload-encoding-contract.md
唯一事实源判据: swarm/db/proposal_yaml_parser.py::is_mojibake_text
  —— 本文件 vendored 了一份副本, 并由 tests/unit/test_encoding_selfcheck.py
     逐条断言两者判定一致 (零口径漂移)。

设计要点:
  * `probe` 给的是**本机环境的真结论**, 不是抽象形状表 —— 同一段代码在
    `LANG=zh_CN.UTF-8` 与 `LANG=C` 下的结论完全不同, 必须实测。
  * `static` 按"目标文件当前是否含非 ASCII 字节"分三档 (高危/潜伏/未知),
    避免把 32 处 shape-1 撒成同等优先级的告警。
  * `mojibake` 用与仓内相同的判据扫存量, 命中即给 C1 控制符计数。
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import locale
import os
import re
import sys
import tempfile
from pathlib import Path

__version__ = "1.0"

# ---------------------------------------------------------------------------
# vendored 判据 (与 swarm/db/proposal_yaml_parser.py 逐条一致)
#
# 判据 (三重条件, 缺一不可):
#   ① 文本中"字符全部 <= U+00FF"的连续段 (latin-1 可表示) 且长度 >= 4;
#   ② 该段含 >=1 个 C1 控制符 (U+0080-U+009F) —— 合法文本几乎不含 C1,
#      而 UTF-8 续字节 0x80-0x9F 经 latin-1 误读恰好落进 C1 区;
#   ③ 该段能按 UTF-8 **增量**解码出 >=80% 的字符 (段尾不完整序列只解出
#      其之前部分, 残余字符逐字保留 —— 不猜、不丢)。
# 三条同时满足才判 mojibake ⇒ 真实重音字符 (café / naïve / ü) 与
# emoji / 中文零误判; 不可判定 (往返失败 / 无 C1 / 可解比例不足) ⇒ 不猜, 只报不修。
# ---------------------------------------------------------------------------

_C1_START = 0x80
_C1_END = 0x9F
_MOJI_MIN_RUN = 4


def _latin1_runs(text: str) -> list[tuple[int, int]]:
    """切出 text 中所有"字符全部 <= U+00FF"的最大连续段 (下标区间)。"""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if ord(ch) <= 0xFF:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(text)))
    return runs


def _c1_count(text: str) -> int:
    """C1 控制符 (U+0080-U+009F) 计数 —— 误读的最强信号。"""
    return sum(1 for c in text if _C1_START <= ord(c) <= _C1_END)


def _decode_latin1_run(run: str) -> tuple[str, int]:
    """把一段 latin-1 可表示文本按 UTF-8 **增量**解码。

    Returns:
        (可解出的文本, 消耗的字符数)。段尾若是不完整序列, 只解出其之前的
        部分, 剩余字符由调用方**逐字保留** —— 不猜、不丢。
    """
    raw = run.encode("latin-1")
    try:
        return raw.decode("utf-8"), len(run)
    except UnicodeDecodeError as exc:
        prefix = raw[: exc.start]
        try:
            return prefix.decode("utf-8"), len(prefix)
        except UnicodeDecodeError:
            return "", 0


def _run_mojibake_verdict(run: str) -> tuple[bool, str, int]:
    """判定一段文本并给出可无损修复的部分。

    Returns:
        (是否 mojibake, 可解出的修复文本, 消耗的字符数)。
    """
    if len(run) < _MOJI_MIN_RUN:
        return False, "", 0
    if not _c1_count(run):
        return False, "", 0
    fixed, consumed = _decode_latin1_run(run)
    if consumed < _MOJI_MIN_RUN:
        return False, "", 0
    # 至少 80% 可无损解码才算误读 (短前缀解码成功属偶然)
    if consumed * 10 < len(run) * 8:
        return False, "", 0
    # 修复后 C1 控制符必须显著减少 (合法文本几乎不含 C1)
    if _c1_count(fixed) >= _c1_count(run):
        return False, "", 0
    return True, fixed, consumed


def is_mojibake_text(text: str | None) -> bool:
    """文本是否含 mojibake 片段 (UTF-8 被 latin-1 误读)。"""
    if not text:
        return False
    return any(_run_mojibake_verdict(text[a:b])[0] for a, b in _latin1_runs(text))


def mojibake_spans(text: str | None) -> list[tuple[int, int]]:
    """定位文本中的 mojibake 片段 (下标区间)。"""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    for a, b in _latin1_runs(text):
        ok, _fixed, consumed = _run_mojibake_verdict(text[a:b])
        if ok:
            spans.append((a, a + consumed))
    return spans


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 探针载荷 —— 中文 + 重音 + emoji (三类都要: 中文验多字节, 重音验 latin-1 可表示,
#: emoji 验 4 字节序列)。真实重音是**负样本**保护: 它不该被判成 mojibake。
PROBE_TEXT = "编码契约自检探针 — café naïve ünïcode 🚀 中文载荷"

#: 默认排除目录 (与仓内 env-lint 同思路: 排除 vendored / 测试 / 虚拟环境)。
#: 以 `*` 结尾的条目按**前缀**匹配 (如 `generation-*` 命中 `generation-1785664750-xxx`)。
DEFAULT_EXCLUDES = (
    "site-packages", "node_modules", ".venv*", "venv*", "env", ".git",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".tox", "vendor", "third_party",
    ".hermes-runtime", ".tmp", "generation-*", "cpython-*", "python-*",
    # 仓内 worktree 副本树 (每个都是全量仓库副本 ⇒ 扫描量爆炸, 且重复计入命中)
    ".worktrees", ".claude", ".swarm", ".agents",
)

#: 数据文件扩展名 (mojibake 审计面)
DATA_EXTS = (".json", ".md", ".yaml", ".yml", ".txt", ".jsonl", ".toml", ".ini")

#: **字节级预筛** —— C1 控制符 (U+0080-U+009F) 在 UTF-8 里必为 2 字节 `C2 80`–`C2 9F`。
#: 文件若不含该字节序列 ⇒ 不可能含 C1 字符 ⇒ 判据三重条件的第②条必然不满足
#: ⇒ **不可能**是 mojibake ⇒ 直接跳过，无需跑逐字符判据。
#: 这是**无损剪枝**（零假阴性），把大目录扫描从"逐字符 Python 循环"降到"C 速字节匹配"，
#: 实测在 8.2G 会话转录树上把单遍耗时从 >20 分钟压到分钟级。
_C1_UTF8_BYTES = re.compile(rb"[\xc2][\x80-\x9f]")

SEV_HIGH = "high"      # 高危: 目标文件当前含非 ASCII ⇒ locale 非 UTF-8 即损坏
SEV_LATENT = "latent"  # 潜伏: 目标文件当前纯 ASCII ⇒ 今天无害, 加中文即变高危
SEV_UNKNOWN = "unknown"  # 未知: 路径动态拼装 ⇒ 需人工判读
SEV_PAYLOAD = "payload"  # 载荷入口: 文本模式 stdin 读 ⇒ **静默乱码**风险 (活险)
SEV_LOUD = "loud"      # 响亮失败: 文本模式 stdout/stderr 写 ⇒ UnicodeEncodeError

#: 分级排序 (越靠前越该先治)
SEV_ORDER = (SEV_HIGH, SEV_PAYLOAD, SEV_LATENT, SEV_UNKNOWN, SEV_LOUD)


# ---------------------------------------------------------------------------
# 模式 1: probe —— 本机环境实测
# ---------------------------------------------------------------------------

def _stdin_encoding() -> str:
    enc = getattr(sys.stdin, "encoding", None)
    if enc:
        return enc
    try:
        return locale.getpreferredencoding(False)
    except Exception:
        return "unknown"


def _stdout_encoding() -> str:
    enc = getattr(sys.stdout, "encoding", None)
    if enc:
        return enc
    try:
        return locale.getpreferredencoding(False)
    except Exception:
        return "unknown"


def cmd_probe(args: argparse.Namespace) -> int:
    """实测**本机环境**的载荷编码行为, 给出"当前安全 / 当前危险"的结论。"""
    fs_enc = sys.getfilesystemencoding()
    try:
        pref_enc = locale.getpreferredencoding(False)
    except Exception:
        pref_enc = "unknown"
    stdin_enc = _stdin_encoding()
    stdout_enc = _stdout_encoding()

    env_bits = {
        "LANG": os.environ.get("LANG", ""),
        "LC_ALL": os.environ.get("LC_ALL", ""),
        "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", ""),
        "PYTHONUTF8": os.environ.get("PYTHONUTF8", ""),
    }

    # --- 文件面 (形状 1): 探针文件用**无 encoding** 的默认行为读回 ------------
    tmpdir = tempfile.mkdtemp(prefix="enc-selfcheck-")
    probe_path = Path(tmpdir) / "probe.txt"
    probe_path.write_text(PROBE_TEXT, encoding="utf-8")   # 写入一律显式 UTF-8
    try:
        read_back = probe_path.read_text()                 # ← 故意不传 encoding
        file_roundtrip_ok = read_back == PROBE_TEXT
        read_err = None
    except Exception as exc:                              # pragma: no cover
        read_back = ""
        file_roundtrip_ok = False
        read_err = f"{type(exc).__name__}: {exc}"

    # --- 载荷面 (形状 3): UTF-8 字节流按**环境声明的编码**误解码 ---------------
    raw = PROBE_TEXT.encode("utf-8")
    try:
        misdecoded = raw.decode(stdin_enc)
        shape3_broken = is_mojibake_text(misdecoded)
        shape3_err = None
    except (UnicodeDecodeError, LookupError) as exc:
        misdecoded = ""
        shape3_broken = True
        shape3_err = f"{type(exc).__name__}: {exc}"

    # --- 输出面 (形状 3 的另一半): stdout 写中文会不会炸 ----------------------
    # 这是 08:46 事故的**第二个**故障模式 (MCP stdout 写中文抛 UnicodeEncodeError);
    # 与"静默乱码"不同 —— 它是**响亮失败**, 但同样会打断链路。
    # 这里**忠实模拟** `sys.stdout.write(str)` 的编码动作 (TextIOWrapper 就是用
    # stream.encoding + stream.errors 编码的), 不实际写, 以免污染输出。
    stdout_errors = getattr(sys.stdout, "errors", None) or "strict"
    try:
        PROBE_TEXT.encode(stdout_enc, errors=stdout_errors)
        stdout_ok = True
        stdout_err = None
    except (UnicodeEncodeError, LookupError) as exc:
        stdout_ok = False
        stdout_err = f"{type(exc).__name__}: {exc}"

    safe = bool(file_roundtrip_ok) and not shape3_broken and stdout_ok

    result = {
        "version": __version__,
        "mode": "probe",
        "encodings": {
            "filesystem": fs_enc,
            "preferred": pref_enc,
            "stdin": stdin_enc,
            "stdout": stdout_enc,
        },
        "env": env_bits,
        "file_roundtrip": {
            "ok": bool(file_roundtrip_ok),
            "error": read_err,
            "sample": read_back[:60],
        },
        "payload_misdecode": {
            "encoding": stdin_enc,
            "mojibake": bool(shape3_broken),
            "error": shape3_err,
            "sample": misdecoded[:60],
        },
        "stdout_writable": {
            "ok": bool(stdout_ok),
            "error": stdout_err,
        },
        "verdict": "safe" if safe else "unsafe",
        "verdict_zh": "当前环境安全" if safe else "当前环境危险",
        "note": (
            "本结论只针对**当前环境**。无 encoding 的读改写在 locale 漂移的启动环境"
            "(cron / systemd / 容器 / IDE 面板)下会重新变成危险 —— 契约规则 1/2/4 要求"
            "**显式** encoding, 不依赖本环境恰好是 UTF-8。"
        ),
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("编码契约自检 · probe (本机环境实测)")
        print(f"  filesystem encoding : {fs_enc}")
        print(f"  preferred encoding : {pref_enc}")
        print(f"  stdin encoding      : {stdin_enc}")
        print(f"  stdout encoding     : {stdout_enc}")
        print(f"  env                 : LANG={env_bits['LANG']!r} "
              f"LC_ALL={env_bits['LC_ALL']!r} "
              f"PYTHONIOENCODING={env_bits['PYTHONIOENCODING']!r}")
        print(f"  文件面 (形状1) 无 encoding 读回 : "
              f"{'✅ 逐字一致' if result['file_roundtrip']['ok'] else '❌ 已损坏'}"
              + (f"  ({result['file_roundtrip']['error']})" if result["file_roundtrip"]["error"] else ""))
        print(f"  载荷面 (形状3) UTF-8 字节误解码 : "
              f"{'✅ 安全' if not shape3_broken else '❌ 会乱码'}"
              + (f"  ({shape3_err})" if shape3_err else ""))
        print(f"  输出面 (形状3) stdout 写中文     : "
              f"{'✅ 可写' if stdout_ok else '❌ 抛异常'}"
              + (f"  ({stdout_err})" if stdout_err else ""))
        print(f"⇒ 判定: {result['verdict_zh']}")
        print(f"  提醒: {result['note']}")

    try:
        probe_path.unlink()
        os.rmdir(tmpdir)
    except OSError:
        pass

    return 0 if safe else 1


# ---------------------------------------------------------------------------
# 模式 2: static —— 自身读改写点扫描
# ---------------------------------------------------------------------------

class _ShapeScanner(ast.NodeVisitor):
    """扫 shape-1 / shape-3 调用点。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[dict] = []
        self._func_stack: list[str] = []

    # -- 作用域追踪: 给出命中点所在函数, 便于人工判读 "未知" 档 ------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    @property
    def _scope(self) -> str:
        return self._func_stack[-1] if self._func_stack else "<module>"

    # -- shape-1: Path.read_text() 无 encoding --------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "read_text"
                and not any(k.arg == "encoding" for k in node.keywords)):
                self.findings.append({
                    "kind": "shape1",
                    "line": node.lineno,
                    "call": f"{ast.unparse(func)}()",
                    "scope": self._scope,
                    "literal_path": self._literal_path(func.value),
                })
        # -- shape-1: open(...).read() 且 open 未给 encoding -------------------
        if (isinstance(func, ast.Attribute) and func.attr in ("read", "readlines")):
            owner = func.value
            if (isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name)
                    and owner.func.id == "open"
                    and not any(k.arg == "encoding" for k in owner.keywords)):
                self.findings.append({
                        "kind": "shape1",
                        "line": node.lineno,
                        "call": f"open(...).{func.attr}()",
                        "scope": self._scope,
                        "literal_path": self._literal_path(owner.args[0] if owner.args else None),
                    })
        # -- shape-3-in: 文本模式 sys.stdin 读 —— **静默乱码**风险 (载荷入口) ----
        if (isinstance(func, ast.Attribute) and func.attr in ("read", "readline", "readlines")
                and self._is_sys_attr(func.value, "stdin")):
                self.findings.append({
                    "kind": "shape3-in",
                    "line": node.lineno,
                    "call": f"sys.stdin.{func.attr}()",
                    "scope": self._scope,
                    "literal_path": None,
                })
        # -- shape-3-out: sys.stdout/stderr 写 —— **响亮失败**风险 (UnicodeEncodeError)
        if isinstance(func, ast.Attribute) and func.attr == "write":
            for _name in ("stdout", "stderr"):
                if self._is_sys_attr(func.value, _name):
                    self.findings.append({
                        "kind": "shape3-out",
                        "line": node.lineno,
                        "call": f"sys.{_name}.write()",
                        "scope": self._scope,
                        "literal_path": None,
                    })
        self.generic_visit(node)

    @staticmethod
    def _is_sys_attr(node: ast.AST, attr: str) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr == attr
                and isinstance(node.value, ast.Name) and node.value.id == "sys")

    @staticmethod
    def _literal_path(node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "Path" and node.args:
            return _ShapeScanner._literal_path(node.args[0])
        return None


def _classify_target(literal: str | None, src_file: Path, root: Path) -> tuple[str, str]:
    """按"目标文件当前是否含非 ASCII 字节"分级。

    Returns:
        (严重度, 判定依据文字)。
    """
    if not literal:
        return SEV_UNKNOWN, "路径为动态表达式, 需人工判读"
    candidates = []
    p = Path(literal)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(src_file.parent / p)
        candidates.append(root / p)
    for cand in candidates:
        try:
            if cand.is_file():
                raw = cand.read_bytes()
                non_ascii = sum(1 for b in raw if b > 0x7F)
                if non_ascii:
                    return SEV_HIGH, f"目标含 {non_ascii} 个非 ASCII 字节 (locale 非 UTF-8 即损坏)"
                return SEV_LATENT, "目标当前纯 ASCII, 今天无害, 一旦写入非 ASCII 即变高危"
        except OSError:
            continue
    return SEV_UNKNOWN, "字面量路径在扫描根下不存在"


def _is_excluded(name: str, excludes: tuple[str, ...]) -> bool:
    """排除判定 —— 以 `*` 结尾的条目按**前缀**匹配。"""
    for pat in excludes:
        if pat.endswith("*"):
            if name.startswith(pat[:-1]):
                return True
        elif name == pat:
            return True
    return False


# ---------------------------------------------------------------------------
# baseline —— 只减不增的门禁语义
# ---------------------------------------------------------------------------

#: 门禁默认关注的分级 (其余分级不参与失败判定, 避免噪音淹没真问题):
#:   high    —— 字面量路径指向含非 ASCII 的文件 ⇒ locale 非 UTF-8 即损坏
#:   payload —— 文本模式 stdin 读 ⇒ **静默乱码**风险 (活险)
#: `loud` (stdout/stderr 写 ⇒ UnicodeEncodeError) 是**响亮失败**, 默认不参与门禁,
#: 除非显式 `--fail-on loud`。
DEFAULT_FAIL_ON = "high,payload"

ALL_SEVERITIES = (SEV_HIGH, SEV_PAYLOAD, SEV_LATENT, SEV_UNKNOWN, SEV_LOUD)


def load_baseline(path: str | None) -> set[str]:
    """读基线文件 —— 每行一个命中指纹, `#` 开头为注释。"""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    out: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def write_baseline(path: str, fingerprints: set[str], header: str = "") -> None:
    """写基线文件 —— 排序稳定, 便于 diff 与 code review。"""
    lines: list[str] = []
    if header:
        lines.extend(f"# {ln}" for ln in header.splitlines())
    lines.extend(sorted(fingerprints))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fingerprint(rel: str, kind: str, call: str) -> str:
    """命中指纹 —— **行号无关**。

    早期版本用 `rel:line:kind`, 但行号会随无关代码增删而漂移: 上游持续演进时
    存量命中会被误判为"新增" ⇒ 门禁恒红 ⇒ 基线迅速腐烂。
    改用 `rel:kind:call` (文件 + 形态 + 调用表达式): 行号漂移不影响判定,
    而同一文件里出现**新的调用形态**仍能被识别为新增。
    """
    call_norm = " ".join((call or "").split())
    return f"{rel}:{kind}:{call_norm}"


def parse_fail_on(spec: str | None) -> set[str]:
    """解析 `--fail-on` 分级列表。"""
    if not spec:
        return set(DEFAULT_FAIL_ON.split(","))
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    unknown = wanted - set(ALL_SEVERITIES)
    if unknown:
        raise SystemExit(f"--fail-on 含未知分级: {sorted(unknown)} (可选: {', '.join(ALL_SEVERITIES)})")
    return wanted


def cmd_static(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    excludes = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude or ())
    include_tests = bool(args.include_tests)
    fail_on = parse_fail_on(args.fail_on)
    baseline = load_baseline(args.baseline)

    findings: list[dict] = []
    scanned = 0
    parse_errors = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _is_excluded(d, excludes)]
        if not include_tests:
            dirnames[:] = [d for d in dirnames if d not in ("tests", "test")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fpath = Path(dirpath) / fn
            try:
                tree = ast.parse(fpath.read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError):
                parse_errors += 1
                continue
            scanned += 1
            scanner = _ShapeScanner(fpath)
            scanner.visit(tree)
            for f in scanner.findings:
                if f["kind"] == "shape3-in":
                    sev, why = SEV_PAYLOAD, "文本模式 stdin 读 ⇒ 解码口径 = locale ⇒ 静默乱码风险"
                elif f["kind"] == "shape3-out":
                    sev, why = SEV_LOUD, "文本模式 stdout/stderr 写 ⇒ 非 UTF-8 环境抛 UnicodeEncodeError (响亮失败)"
                else:
                    sev, why = _classify_target(f["literal_path"], fpath, root)
                findings.append({
                    "kind": f["kind"],
                    "severity": sev,
                    "reason": why,
                    "file": str(fpath),
                    "rel": str(fpath.relative_to(root)),
                    "line": f["line"],
                    "call": f["call"],
                    "scope": f.get("scope", "<module>"),
                    "literal_path": f["literal_path"],
                })

    order = {sev: i for i, sev in enumerate(SEV_ORDER)}
    findings.sort(key=lambda f: (order[f["severity"]], f["kind"], f["rel"], f["line"]))

    counts = {
        "high": sum(1 for f in findings if f["severity"] == SEV_HIGH),
        "latent": sum(1 for f in findings if f["severity"] == SEV_LATENT),
        "unknown": sum(1 for f in findings if f["severity"] == SEV_UNKNOWN),
        "payload": sum(1 for f in findings if f["severity"] == SEV_PAYLOAD),
        "loud": sum(1 for f in findings if f["severity"] == SEV_LOUD),
        "shape1": sum(1 for f in findings if f["kind"] == "shape1"),
        "shape3_in": sum(1 for f in findings if f["kind"] == "shape3-in"),
        "shape3_out": sum(1 for f in findings if f["kind"] == "shape3-out"),
    }

    result = {
        "version": __version__,
        "mode": "static",
        "root": str(root),
        "files_scanned": scanned,
        "parse_errors": parse_errors,
        "counts": counts,
        "findings": findings,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"编码契约自检 · static (扫描根: {root})")
        print(f"  扫描 .py 文件      : {scanned}  (解析失败 {parse_errors})")
        print(f"  命中总数           : {len(findings)}"
              f"  (shape1={counts['shape1']} shape3-in={counts['shape3_in']} shape3-out={counts['shape3_out']})")
        print(f"  shape1 分级        : 高危={counts['high']} 潜伏={counts['latent']} 未知={counts['unknown']}")
        print(f"  载荷入口(shape3-in): {counts['payload']} 处  ← 静默乱码风险")
        print(f"  输出面(shape3-out) : {counts['loud']} 处  ← UnicodeEncodeError 响亮失败")
        print("  ── 高危明细 (优先治这些) ──")
        shown = 0
        for f in findings:
            if f["severity"] != SEV_HIGH:
                continue
            shown += 1
            print(f"   [{f['kind']}] {f['rel']}:{f['line']}  {f['call']}  — {f['reason']}")
            if shown >= args.limit:
                print("   … 其余高危项见 --json 输出")
                break
        if not shown:
            print("   (无)")
        if counts["shape3_in"]:
            print("  ── shape-3-in 载荷入口 (文本模式 stdin 读 ⇒ 静默乱码风险, 全部列出) ──")
            for f in findings:
                if f["kind"] == "shape3-in":
                    print(f"   {f['rel']}:{f['line']}  {f['call']}  (in {f['scope']})")
        if counts["shape3_out"]:
            print(f"  ── shape-3-out 输出面 {counts['shape3_out']} 处 (文本模式 stdout/stderr 写 ⇒ "
                  f"UnicodeEncodeError 响亮失败, 非静默损坏; 详见 --json) ──")

    # ---- baseline 门禁判定 (只减不增) ----------------------------------------
    fingerprints = {fingerprint(f["rel"], f["kind"], f["call"]) for f in findings}
    if args.write_baseline:
        write_baseline(
            args.write_baseline,
            fingerprints,
            header=("编码自检 baseline (只减不增) —— 每行一个命中指纹 rel:line:kind\n"
                    f"生成于: {args.root}\n"
                    "语义: 指纹在此 ⇒ 存量, 不红; 指纹不在此 ⇒ 新增 ⇒ 门禁红"),
        )
        if not args.json:
            print(f"  baseline 已写入 : {args.write_baseline} ({len(fingerprints)} 条)")
        return 0

    gated = [f for f in findings if f["severity"] in fail_on]
    new_ones = [f for f in gated
                if fingerprint(f["rel"], f["kind"], f["call"]) not in baseline]
    result["baseline_size"] = len(baseline)
    result["gated_findings"] = len(gated)
    result["new_findings"] = [
        {"rel": f["rel"], "line": f["line"], "kind": f["kind"], "severity": f["severity"],
         "call": f["call"], "scope": f["scope"]} for f in new_ones
    ]

    if new_ones:
        if not args.json:
            print(f"  ── 🔴 新增命中 {len(new_ones)} 处 (不在 baseline 内 ⇒ 门禁红) ──")
            for f in new_ones[:args.limit]:
                print(f"   [{f['severity']}] {f['rel']}:{f['line']}  {f['call']}  (in {f['scope']})")
            if len(new_ones) > args.limit:
                print(f"   … 其余 {len(new_ones) - args.limit} 处见 --json 输出")
        return 1

    if not args.json:
        print(f"  baseline 判定     : 新增 0 处 ⇒ 绿 (存量 {len(baseline)} 条已在基线内)")
    return 0


# ---------------------------------------------------------------------------
# 模式 3: mojibake —— 存量审计
# ---------------------------------------------------------------------------

def cmd_mojibake(args: argparse.Namespace) -> int:
    excludes = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude or ())
    scanned = 0
    hits: list[dict] = []
    unreadable = 0
    prefiltered = 0

    for target in args.paths:
        root = Path(target).resolve()
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not _is_excluded(d, excludes)]
                for fn in filenames:
                    if Path(fn).suffix.lower() in DATA_EXTS:
                        files.append(Path(dirpath) / fn)
        else:
            continue
        for fpath in files:
            try:
                raw = fpath.read_bytes()
            except OSError:
                unreadable += 1
                continue
            scanned += 1
            # 字节级预筛: 不含 C2 80-9F ⇒ 不可能含 C1 字符 ⇒ 判据第②条必不满足
            # ⇒ 不可能是 mojibake ⇒ 无需跑逐字符判据 (无损剪枝, 零假阴性)
            if not _C1_UTF8_BYTES.search(raw):
                prefiltered += 1
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                unreadable += 1
                continue
            if is_mojibake_text(text):
                spans = mojibake_spans(text)
                c1 = _c1_count(text)
                try:
                    rel = str(fpath.relative_to(root)) if root.is_dir() else fpath.name
                except ValueError:  # pragma: no cover
                    rel = fpath.name
                hits.append({
                    "file": str(fpath),
                    "rel": rel,
                    "length": len(text),
                    "c1_count": c1,
                    "span_count": len(spans),
                    "sample": text[spans[0][0]:spans[0][1]][:80] if spans else "",
                })

    result = {
        "version": __version__,
        "mode": "mojibake",
        "files_scanned": scanned,
        "prefiltered": prefiltered,
        "judged": scanned - prefiltered - unreadable,
        "unreadable": unreadable,
        "hits": hits,
        "hit_count": len(hits),
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("编码契约自检 · mojibake 存量审计")
        print(f"  扫描文件数 : {scanned}   (字节预筛跳过 {prefiltered} / 非 UTF-8 跳过 {unreadable})")
        print(f"  实跑判据   : {scanned - prefiltered - unreadable} 个文件")
        print(f"  mojibake   : {len(hits)} 命中")
        for h in hits[:args.limit]:
            print(f"   ❌ C1={h['c1_count']:6d} len={h['length']:8d} {h['file']}")
            if h["sample"]:
                print(f"        片段: {h['sample']!r}")
        if len(hits) > args.limit:
            print(f"   … 其余 {len(hits) - args.limit} 项见 --json 输出")

    # ---- baseline 门禁判定 (只减不增) ----------------------------------------
    fingerprints = {h["rel"] for h in hits}
    if args.write_baseline:
        write_baseline(
            args.write_baseline,
            fingerprints,
            header=("编码自检 baseline (只减不增) —— 每行一个命中文件相对路径\n"
                    "语义: 路径在此 ⇒ 存量, 不红; 不在此 ⇒ 新增 ⇒ 门禁红"),
        )
        if not args.json:
            print(f"  baseline 已写入 : {args.write_baseline} ({len(fingerprints)} 条)")
        return 0

    baseline = load_baseline(args.baseline)
    new_ones = [h for h in hits if h["rel"] not in baseline]
    result["baseline_size"] = len(baseline)
    result["new_hits"] = [h["rel"] for h in new_ones]
    if new_ones:
        if not args.json:
            print(f"  ── 🔴 新增命中 {len(new_ones)} 处 (不在 baseline 内 ⇒ 门禁红) ──")
            for h in new_ones[:args.limit]:
                print(f"   ❌ C1={h['c1_count']:6d} {h['rel']}")
        return 1
    if not args.json:
        print(f"  baseline 判定     : 新增 0 处 ⇒ 绿 (存量 {len(baseline)} 条已在基线内)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _force_utf8_stdio() -> bool:
    """把 stdout/stderr 强制成 UTF-8 (不依赖环境 locale)。

    **套件自身不能踩它所检测的坑**: 在 `LC_ALL=C` / `PYTHONIOENCODING=latin-1`
    环境里, 直接 print 中文会抛 `UnicodeEncodeError` —— 正是契约形状 3 的
    现场表现。故本套件的**所有**输出都走显式 UTF-8 字节流。

    Returns:
        是否成功 reconfigure (False ⇒ 回退到 errors="replace" 尽力输出)。
    """
    ok = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")
            ok = True
    return ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="encoding_selfcheck.py",
        description="编码契约自检套件 (独立可分发, 零依赖本仓)",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("probe", help="本机环境实测 (编码 + 真实往返)")
    pp.add_argument("--json", action="store_true", help="机读输出")
    pp.set_defaults(func=cmd_probe)

    ps = sub.add_parser("static", help="扫自身 shape-1 / shape-3 读改写点")
    ps.add_argument("root", help="扫描根目录")
    ps.add_argument("--exclude", action="append", help="额外排除目录名 (可重复)")
    ps.add_argument("--include-tests", action="store_true", help="不排除 tests/ 目录")
    ps.add_argument("--limit", type=int, default=40, help="终端明细最多打印条数")
    ps.add_argument("--json", action="store_true", help="机读输出")
    ps.add_argument("--baseline", help="baseline 文件 —— 指纹在此=存量不红, 不在此=新增即红 (只减不增)")
    ps.add_argument("--write-baseline", help="写入 baseline 文件并退出 (生成/刷新)")
    ps.add_argument("--fail-on", default=DEFAULT_FAIL_ON,
                    help=f"参与失败判定的分级 (默认 {DEFAULT_FAIL_ON}); 可选: {','.join(ALL_SEVERITIES)}")
    ps.set_defaults(func=cmd_static)

    pm = sub.add_parser("mojibake", help="扫存量乱码")
    pm.add_argument("paths", nargs="+", help="文件或目录")
    pm.add_argument("--exclude", action="append", help="额外排除目录名 (可重复)")
    pm.add_argument("--limit", type=int, default=20, help="终端明细最多打印条数")
    pm.add_argument("--json", action="store_true", help="机读输出")
    pm.add_argument("--baseline", help="baseline 文件 —— 相对路径在此=存量不红, 不在此=新增即红")
    pm.add_argument("--write-baseline", help="写入 baseline 文件并退出 (生成/刷新)")
    pm.set_defaults(func=cmd_mojibake)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _force_utf8_stdio()
    return int(args.func(args))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)
