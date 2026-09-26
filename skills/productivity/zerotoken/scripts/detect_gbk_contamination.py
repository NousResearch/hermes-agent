"""
detect_gbk_contamination.py — 检测并修复 UTF-8 文件中的 GBK 编码污染

解决的问题：
- PowerShell Add-Content 使用 GBK 编码追加中文字符到 UTF-8 文件
- 文件变成 UTF-8 + GBK 混合编码，中文显示为乱码
- 常规检测工具（如 fix_encoding.py scan）将这类文件标记为 "unknown"
  但无法区分"纯 GBK 文件"和"UTF-8 被 GBK 追加污染的文件"

原理：
    UTF-8 多字节序列的起始字节必须是 0xC0-0xFD，后续字节必须是 0x80-0xBF。
    GBK 双字节编码的首字节范围是 0x81-0xFE（与 UTF-8 首字节范围不重叠），
    次字节范围是 0x40-0xFE（不含 0x7F）。
    当 GBK 内容被追加到 UTF-8 文件尾部时，污染区域的第一个字节通常是 0x81-0xFE
    范围内的值，这不会出现在合法的 UTF-8 多字节序列起始位置。
    本脚本通过这个特征定位污染边界并修复。

用法：
    # 扫描目录，报告被 GBK 污染的文件
    python scripts/detect_gbk_contamination.py scan <directory>
    python scripts/detect_gbk_contamination.py scan <directory> --ext .md,.txt

    # 查看单个文件的污染详情
    python scripts/detect_gbk_contamination.py inspect <file>

    # 修复被污染的文件
    python scripts/detect_gbk_contamination.py fix <directory> --backup
    python scripts/detect_gbk_contamination.py fix <directory> --ext .md --backup

    # 仅预览（不写文件）
    python scripts/detect_gbk_contamination.py fix <directory> --preview

安全保证：
    - fix 模式需要 --backup 参数才会创建 .bak 备份
    - preview 不做任何写入
    - 只修改匹配 --ext 的文件
"""

import argparse
import glob
import os
import shutil
import sys
from typing import List, Optional, Tuple


# ── 安全打印（解决 #6: GBK 控制台 UnicodeEncodeError）──
# 规范 2/15：不依赖系统默认字符集，显式把 stdio 切到 UTF-8
# （Python 3.7+；旧版本或不可重配置流保持原样）
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError, ValueError):
    pass

_STDOUT_ENCODING = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'


def sp(*args, **kwargs) -> None:
    """safe_print：GBK 环境不崩溃。"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        sa = [a.encode(_STDOUT_ENCODING, errors='replace').decode(
              _STDOUT_ENCODING, errors='replace')
              if isinstance(a, str) else str(a) for a in args]
        print(*sa, **kwargs)


# ── 核心检测算法 ──────────────────────────────────────────


def find_first_invalid_utf8_byte(raw: bytes) -> int:
    """找到第一个导致 UTF-8 解码失败的字节偏移。

    返回 -1 表示整个文件是合法的 UTF-8。
    """
    i = 0
    length = len(raw)
    while i < length:
        b = raw[i]
        if b < 0x80:
            # ASCII 单字节 (0x00-0x7F)
            i += 1
        elif 0xC2 <= b <= 0xDF:
            # 2 字节 UTF-8: 110xxxxx 10xxxxxx
            if i + 1 >= length or not (0x80 <= raw[i + 1] <= 0xBF):
                return i
            i += 2
        elif 0xE0 <= b <= 0xEF:
            # 3 字节 UTF-8: 1110xxxx 10xxxxxx 10xxxxxx
            if i + 2 >= length:
                return i
            if not (0x80 <= raw[i + 1] <= 0xBF) or not (0x80 <= raw[i + 2] <= 0xBF):
                return i
            # 检查超长编码
            if b == 0xE0 and raw[i + 1] < 0xA0:
                return i
            if b == 0xED and raw[i + 1] > 0x9F:
                return i
            i += 3
        elif 0xF0 <= b <= 0xF4:
            # 4 字节 UTF-8: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            if i + 3 >= length:
                return i
            if (not (0x80 <= raw[i + 1] <= 0xBF) or
                    not (0x80 <= raw[i + 2] <= 0xBF) or
                    not (0x80 <= raw[i + 3] <= 0xBF)):
                return i
            if b == 0xF0 and raw[i + 1] < 0x90:
                return i
            if b == 0xF4 and raw[i + 1] > 0x8F:
                return i
            i += 4
        else:
            # 0x80-0xBF（孤立的后续字节）或 0xC0-0xC1（过长的 2 字节编码）或 0xF5-0xFF
            # 这些都不是合法的 UTF-8 起始字节
            # GBK 首字节范围 0x81-0xFE 大部分落在这里
            return i
    return -1


def is_gbk_contaminated(raw: bytes) -> Tuple[bool, int, Optional[str]]:
    """检测文件是否被 GBK 编码污染。

    Returns:
        (是否被污染, 第一个无效字节偏移, 从该位置开始的 GBK 解码片段(前50字符))
    """
    pos = find_first_invalid_utf8_byte(raw)
    if pos == -1:
        return False, -1, None

    # 从无效位置开始检查是否为有效的 GBK 编码
    gbk_data = raw[pos:]
    try:
        gbk_text = gbk_data.decode('gbk')
        # 成功用 GBK 解码 → 确认是 GBK 污染
        if len(gbk_text) > 0:
            preview = gbk_text[:min(50, len(gbk_text))]
            return True, pos, preview
    except UnicodeDecodeError:
        pass

    # 也可能是其他编码或纯乱码，尝试部分解码
    try:
        gbk_text_partial = gbk_data.decode('gbk', errors='replace')
        # 检查是否有大量有效的 GBK 字符
        valid_chars = sum(1 for c in gbk_text_partial if c != '\ufffd')
        total_chars = len(gbk_text_partial)
        if total_chars > 0 and valid_chars / total_chars > 0.5:
            preview = gbk_text_partial[:min(50, len(gbk_text_partial))]
            return True, pos, preview
    except Exception:
        pass

    return False, pos, None


def repair_contamination(raw: bytes) -> Tuple[str, int, int]:
    """修复 GBK 污染，返回 (修复后的文本, UTF-8 部分字节数, GBK 部分字节数)。

    策略：
    1. 找到第一个无效 UTF-8 字节的位置
    2. 之前的字节作为 UTF-8 解码
    3. 之后的字节作为 GBK 解码
    4. 合并两部分
    """
    pos = find_first_invalid_utf8_byte(raw)
    if pos == -1:
        # 已经是纯 UTF-8
        return raw.decode('utf-8'), len(raw), 0

    # UTF-8 部分（从头到无效位置之前）
    utf8_part = raw[:pos]
    utf8_text = utf8_part.decode('utf-8', errors='replace')

    # GBK 部分（从无效位置开始）
    gbk_data = raw[pos:]
    try:
        gbk_text = gbk_data.decode('gbk')
    except UnicodeDecodeError:
        gbk_text = gbk_data.decode('gbk', errors='replace')

    return utf8_text + gbk_text, len(utf8_part), len(gbk_data)


# ── 文件收集 ──────────────────────────────────────────────


def collect_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """收集需要处理的文本文件列表。"""
    if extensions:
        files = []
        for ext in extensions:
            ext = ext.strip()
            if not ext.startswith('.'):
                ext = '.' + ext
            pattern = os.path.join(directory, '**', f'*{ext}')
            files.extend(glob.glob(pattern, recursive=True))
        return sorted(set(files))
    else:
        # 默认语言文本文件
        text_exts = ['.md', '.txt', '.yaml', '.yml', '.json',
                     '.toml', '.cfg', '.ini', '.conf']
        files = []
        for ext in text_exts:
            pattern = os.path.join(directory, '**', f'*{ext}')
            files.extend(glob.glob(pattern, recursive=True))
        return sorted(set(files))


# ── 操作模式 ──────────────────────────────────────────────


def scan_directory(directory: str, extensions: Optional[List[str]] = None) -> None:
    """扫描目录，报告被 GBK 污染的文件。"""
    files = collect_files(directory, extensions)
    if not files:
        sp(f"在 {directory} 中未找到匹配的文件")
        return

    contaminated = []
    clean_count = 0
    error_count = 0

    for filepath in files:
        if not os.path.isfile(filepath):
            continue

        rel = os.path.relpath(filepath, directory)
        try:
            with open(filepath, 'rb') as f:
                raw = f.read()

            is_contaminated, pos, preview = is_gbk_contaminated(raw)
            if is_contaminated:
                contaminated.append((rel, pos, preview))
                if len(preview) > 40:
                    preview_display = preview[:40] + '...'
                else:
                    preview_display = preview
                sp(f"  [污染] {rel:<55} 偏移={pos:<8} 片段='{preview_display}'")
            else:
                clean_count += 1

        except Exception as e:
            sp(f"  [ERROR] {rel}: {e}")
            error_count += 1

    sp(f"\n总计: {clean_count} 干净, {len(contaminated)} 被污染"
      f"{f', {error_count} 错误' if error_count else ''}")

    if contaminated:
        sp(f"\n建议: 运行 python scripts/detect_gbk_contamination.py fix {directory} "
          f"--backup 修复污染文件")


def inspect_file(filepath: str) -> None:
    """详细查看单个文件的 GBK 污染情况。"""
    if not os.path.isfile(filepath):
        sp(f"错误: 文件不存在: {filepath}")
        return

    abs_path = os.path.abspath(filepath)
    sp(f"=== 检查: {abs_path} ===\n")

    with open(filepath, 'rb') as f:
        raw = f.read()

    file_size = len(raw)
    sp(f"文件大小: {file_size} 字节")

    is_contaminated, pos, preview = is_gbk_contaminated(raw)

    if not is_contaminated:
        if pos == -1:
            sp("状态: ✅ 纯 UTF-8，未被 GBK 污染")
        else:
            sp("状态: ⚠️ 非 UTF-8 编码，但无法确认是 GBK 污染")
        return

    sp(f"状态: ❌ 检测到 GBK 污染")
    sp(f"污染起始位置: 字节偏移 {pos} ({pos / file_size * 100:.1f}%)")
    sp(f"UTF-8 有效部分: {pos} 字节")
    sp(f"  解码片段: {raw[:min(80, pos)].decode('utf-8', errors='replace')[:60]}...")
    sp(f"GBK 污染部分: {file_size - pos} 字节")
    sp(f"  解码片段: {preview}")

    # 显示污染区域的原始字节十六进制
    gbk_start = max(pos - 4, 0)
    gbk_end = min(pos + 40, file_size)
    hex_bytes = ' '.join(f'{b:02x}' for b in raw[gbk_start:gbk_end])
    sp(f"\n污染区域原始字节 (偏移 {gbk_start}-{gbk_end}):")
    sp(f"  {hex_bytes}")

    # 显示 UTF-8 解码失败的上下文
    sp(f"\n污染字节上下文:")
    context_size = 16
    for i in range(pos - context_size, pos + context_size):
        if 0 <= i < file_size:
            marker = " ← 污染开始" if i == pos else ""
            if raw[i] < 0x80:
                char_desc = f"ASCII: {chr(raw[i])}"
            elif 0x81 <= raw[i] <= 0xFE:
                char_desc = f"GBK首字节 {raw[i]:02x}"
                if i + 1 < file_size:
                    char_desc += f" + 0x{raw[i+1]:02x}"
            else:
                char_desc = f"字节 0x{raw[i]:02x}"
            sp(f"  [{i:>8}] {char_desc}{marker}")

    sp(f"\n建议: python scripts/detect_gbk_contamination.py fix \"{filepath}\" --backup")


def fix_contamination(directory_or_file: str, extensions: Optional[List[str]] = None,
                      backup: bool = False, preview: bool = False) -> None:
    """修复被 GBK 污染的文件。"""
    if os.path.isfile(directory_or_file):
        files = [directory_or_file]
        is_single_file = True
    else:
        files = collect_files(directory_or_file, extensions)
        is_single_file = False

    if not files:
        sp(f"未找到匹配的文件")
        return

    fixed = 0
    skipped = 0
    errors = 0

    for filepath in files:
        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, 'rb') as f:
                raw = f.read()

            is_contaminated, pos, preview_text = is_gbk_contaminated(raw)
            if not is_contaminated:
                skipped += 1
                continue

            repaired_text, utf8_len, gbk_len = repair_contamination(raw)

            if is_single_file:
                rel = os.path.basename(filepath)
            else:
                base_dir = directory_or_file if os.path.isdir(directory_or_file) else os.path.dirname(directory_or_file)
                rel = os.path.relpath(filepath, base_dir)

            if preview:
                sp(f"  [预览] {rel:<55} UTF-8部分={utf8_len}B + GBK部分={gbk_len}B")
                sp(f"         修复后前60字符: {repaired_text[:60]}")
            else:
                if backup:
                    bak_path = filepath + '.bak'
                    shutil.copy2(filepath, bak_path)

                with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(repaired_text)

                sp(f"  [修复] {rel:<55} UTF-8部分={utf8_len}B + GBK部分={gbk_len}B")

            fixed += 1

        except Exception as e:
            if is_single_file:
                rel = os.path.basename(filepath)
            else:
                rel = os.path.relpath(filepath, os.path.dirname(directory_or_file)
                                      if os.path.isfile(directory_or_file) else directory_or_file)
            sp(f"  [ERROR] {rel}: {e}")
            errors += 1

    if preview:
        sp(f"\n预览完成: {fixed} 待修复, {skipped} 跳过(无需处理)"
          f"{f', {errors} 错误' if errors else ''}")
        if fixed > 0:
            sp(f"执行修复: python scripts/detect_gbk_contamination.py fix "
              f"\"{directory_or_file}\" --backup")
    else:
        sp(f"\n完成: {fixed} 修复, {skipped} 跳过(无需处理)"
          f"{f', {errors} 错误' if errors else ''}")


# ── CLI ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="检测并修复 UTF-8 文件中的 GBK 编码污染"
    )
    parser.add_argument("mode",
                        choices=["scan", "inspect", "fix"],
                        help="操作模式: scan=扫描目录, inspect=检查单文件, fix=修复")
    parser.add_argument("target", nargs="?", default=".",
                        help="目标目录或文件路径（默认当前目录）")
    parser.add_argument("--ext", default="",
                        help="文件扩展名过滤，逗号分隔，如 .md,.txt,.yaml（仅 scan/fix 模式）")
    parser.add_argument("--backup", action="store_true",
                        help="修复前创建 .bak 备份")
    parser.add_argument("--preview", action="store_true",
                        help="仅预览，不执行写入（仅 fix 模式）")

    args = parser.parse_args()
    extensions = [e.strip() for e in args.ext.split(",")
                  if e.strip()] if args.ext else None

    if args.mode == "scan":
        scan_directory(args.target, extensions)
    elif args.mode == "inspect":
        inspect_file(args.target)
    elif args.mode == "fix":
        fix_contamination(args.target, extensions, args.backup, args.preview)


if __name__ == '__main__':
    main()
