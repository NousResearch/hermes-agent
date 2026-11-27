"""
safe_io.py — 安全文件读写工具模块

解决的问题：
- 文件编码不一致（UTF-8 / UTF-16 / 含损坏字符）
- Python 控制台输出中文在 GBK 环境下失败
- PowerShell Add-Content 用 GBK 编码污染 UTF-8 文件

用法：
    from safe_io import safe_read, safe_write, safe_append, write_result

    content = safe_read("path/to/file.md")
    safe_write("path/to/file.md", content)
    safe_append("path/to/file.md", "追加内容")   # 替代 Add-Content
    write_result("验证通过", out_path="verify_result.txt")
"""

import os
import sys
from typing import Optional


# ── 编码安全的控制台输出 ──────────────────────────────────
# 解决 #6: Python 在 GBK 控制台下 print(中文/特殊符号) 报 UnicodedEncodeError
# 规范 2/15：不依赖系统默认字符集，优先显式把 stdio 切到 UTF-8，
# 而不是把中文替换成 '?'（显示层丢字）。
def _ensure_utf8_stdio() -> None:
    """将 stdout/stderr 显式切换为 UTF-8（Python 3.7+）。

    Windows 中文系统默认控制台代码页为 GBK(936)，不切换时
    print(中文) 会抛 UnicodeEncodeError 或被替换成 '?'。
    切换到 UTF-8 后，配合 `chcp 65001`（或管道重定向）中文可完整显示。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, OSError, ValueError):
            # Python < 3.7 无 reconfigure，或流不支持重配置时保持原样
            pass


_ensure_utf8_stdio()
_STDOUT_ENCODING = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'


def safe_print(*args, **kwargs) -> None:
    """安全打印，自动处理控制台编码不支持 Unicode 的问题。

    模块导入时已尝试把 stdout 切换到 UTF-8；在极少数仍无法
    编码的环境中，将无法编码的字符替换为 '?' 而不是崩溃。
    永远不会有 UnicodeEncodeError。
    """
    try:
        # 先尝试正常 print
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 兜底：对每个参数做编码安全处理
        safe_args = []
        for a in args:
            if isinstance(a, str):
                safe_args.append(a.encode(
                    _STDOUT_ENCODING, errors='replace').decode(
                    _STDOUT_ENCODING, errors='replace'))
            else:
                safe_args.append(str(a))
        print(*safe_args, **kwargs)


def safe_read(path: str) -> str:
    """安全读取 UTF-8、带 BOM 的 UTF-8/UTF-16，及 GB18030 文件。

    优先保留原始文本；仅在所有已知编码都无法解码时才替换损坏字节。
    """
    with open(path, 'rb') as f:
        raw = f.read()

    if raw.startswith(b'\xef\xbb\xbf'):
        return raw.decode('utf-8-sig')

    # 检测 BOM
    if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
        # UTF-16 LE / BE
        try:
            return raw.decode('utf-16')
        except:
            pass

    # UTF-8 优先；GB18030 覆盖 Windows 中文环境中的 GBK/GB2312 文本。
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return raw.decode('gb18030')
        except UnicodeDecodeError:
            return raw.decode('utf-8', errors='replace')


def safe_write(path: str, content: str) -> None:
    """安全写入文件，统一 UTF-8 编码（不含 BOM），行尾统一 LF。"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)


def safe_append(path: str, content: str) -> str:
    """安全追加内容到文件，统一使用 UTF-8 编码。

    专为替代 PowerShell Add-Content 设计。Add-Content 在 Windows 中文版下
    默认使用 GBK 编码写入，会与文件原有的 UTF-8 编码混合导致乱码。

    始终使用 Python open(path, 'a', encoding='utf-8', newline='\n')，确保编码一致。

    Args:
        path: 文件路径（父目录不存在时自动创建）
        content: 要追加的内容（会自动在末尾加换行）

    Returns:
        写入文件的绝对路径

    Example:
        safe_append("笔记.md", "## 新的章节")
        # 等价于 PowerShell 的 Add-Content -Path 笔记.md -Value "## 新的章节"
        # 但使用 UTF-8 而非 GBK
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        f.write(content)
        if not content.endswith('\n'):
            f.write('\n')

    abs_path = os.path.abspath(path)
    safe_print(f"已追加到: {abs_path}")
    return abs_path


def write_result(result: str, out_path: str = "verify_result.txt",
                 append: bool = False) -> str:
    """将结果写入文本文件，避免 print() 在 GBK 环境下报错。

    返回写入的绝对路径，供 read_file 工具读取。
    """
    mode = 'a' if append else 'w'
    with open(out_path, mode, encoding='utf-8', newline='\n') as f:
        f.write(result)
        if not result.endswith('\n'):
            f.write('\n')

    abs_path = os.path.abspath(out_path)
    safe_print(f"结果已写入: {abs_path}  <- 用 read_file 查看")
    return abs_path


def auto_encoding(path: str) -> str:
    """自动判断文件编码并读取。与 safe_read 相同但返回 None 时不会崩溃。

    用于批量脚本需要跳过无法处理的文件时。
    """
    try:
        return safe_read(path)
    except Exception:
        return ""
