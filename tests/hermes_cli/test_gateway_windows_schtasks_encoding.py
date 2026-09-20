"""回归测试: ``_schtasks_encoding()`` 必须给出**真实控制台代码页**, 而不是 locale 的偏好编码。

2026-09-20 实测(中文 Windows):

    PYTHONUTF8=0 -> locale.getpreferredencoding(False) = cp936 | GetOEMCP() = cp936
    PYTHONUTF8=1 -> locale.getpreferredencoding(False) = utf-8 | GetOEMCP() = cp936

而 ``schtasks.exe`` 在两种情况下都吐 OEM 字节(cp936)。开了 UTF-8 模式的进程里用 locale 那个值
去解, 结果不是崩(有 ``errors="replace"``)而是**花屏**: 本地化的
``_FALLBACK_PATTERNS``("access is denied"/"acceso denegado"/"přístup byl odepřen" …)永远匹配不上,
Startup 目录兜底逻辑静默失效 —— 这正是"看着一切正常、其实那条分支从没生效"的那类 bug。

测试对代码页本身不做假设(跑到 cp437/cp936/任何机器上都成立):
用 monkeypatch 把"控制台代码页"换成 936 再断言, 而不是依赖当前机器恰好是 cp936。
"""

from __future__ import annotations

import codecs
import types

import pytest

from hermes_cli import gateway_windows as GW


def test_uses_console_codepage_when_utf8_mode_makes_locale_report_utf8(monkeypatch):
    """核心回归: UTF-8 模式下 locale 说 utf-8, 我们也必须给出 OEM 代码页。"""
    monkeypatch.setattr(GW, "_console_codepage", lambda: 936)
    monkeypatch.setattr(GW.locale, "getpreferredencoding", lambda *a, **k: "utf-8")
    assert GW._schtasks_encoding() == "cp936"


def test_falls_back_to_locale_when_console_codepage_unavailable(monkeypatch):
    """拿不到控制台代码页时才退回 locale(离 Windows / ctypes 不可用的情况)。"""
    monkeypatch.setattr(GW, "_console_codepage", lambda: None)
    monkeypatch.setattr(GW.locale, "getpreferredencoding", lambda *a, **k: "cp1252")
    assert GW._schtasks_encoding() == "cp1252"


def test_falls_back_to_utf8_when_everything_is_unavailable(monkeypatch):
    monkeypatch.setattr(GW, "_console_codepage", lambda: None)

    def boom(*args, **kwargs):
        raise RuntimeError("no locale")

    monkeypatch.setattr(GW.locale, "getpreferredencoding", boom)
    assert GW._schtasks_encoding() == "utf-8"


def test_encoding_round_trips_localized_oem_text(monkeypatch):
    """挑一个真实代码页: 用它解自己编出来的本地化文案必须无损(不能出现替换字符)。"""
    monkeypatch.setattr(GW, "_console_codepage", lambda: 936)
    encoding = GW._schtasks_encoding()
    codecs.lookup(encoding)  # 是真实编解码器
    sample = "错误: 拒绝访问。"
    raw = sample.encode("cp936")
    assert raw.decode(encoding, errors="replace") == sample
    assert "\ufffd" not in raw.decode(encoding, errors="replace")


def test_exec_schtasks_passes_console_encoding_and_errors_replace(monkeypatch):
    """编码之外, 还要确认 subprocess 真的收到了 encoding + errors='replace'。"""
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(GW, "_assert_windows", lambda: None)
    monkeypatch.setattr(GW.shutil, "which", lambda name: r"C:\\Windows\\System32\\schtasks.exe")
    monkeypatch.setattr(GW, "_console_codepage", lambda: 936)
    monkeypatch.setattr(GW.subprocess, "run", fake_run)

    code, out, err = GW._exec_schtasks(["/Query", "/TN", "Hermes_Gateway"])

    assert (code, out, err) == (0, "ok", "")
    assert captured["encoding"] == "cp936"
    assert captured["errors"] == "replace"
    assert captured["text"] is True


def test_exec_schtasks_reports_missing_binary(monkeypatch):
    monkeypatch.setattr(GW, "_assert_windows", lambda: None)
    monkeypatch.setattr(GW.shutil, "which", lambda name: None)
    code, out, err = GW._exec_schtasks(["/Query"])
    assert code == 1
    assert "not found" in err


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "-o", "addopts="]))
