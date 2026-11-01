"""
fix_encoding.py — 批量转换文件编码为 UTF-8

解决的问题：
- 文件编码不一致（UTF-8 / UTF-16 / 混合编码）
- 旧文件中存在因编码损坏产生的替换字符

用法：
    # 扫描目录，显示非 UTF-8 文件
    python scripts/fix_encoding.py scan <directory>
    python scripts/fix_encoding.py scan <directory> --ext .md,.txt

    # 预览将发生的转换（不写文件）
    python scripts/fix_encoding.py preview <directory>

    # 执行转换（备份原始文件）
    python scripts/fix_encoding.py convert <directory> --backup
    python scripts/fix_encoding.py convert <directory> --ext .md,.yaml --backup

    # 检查文件中是否有替换字符
    python scripts/fix_encoding.py check-replacement <directory>

安全保证：
    - convert 模式需要 --backup 参数才会创建 .bak 备份
    - 只修改匹配 --ext 的文件
    - preview 模式不做任何写入
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


def detect_encoding(raw: bytes) -> str:
    """检测文件编码。"""
    if raw.startswith(b'\xff\xfe'):
        return 'utf-16-le'
    if raw.startswith(b'\xfe\xff'):
        return 'utf-16-be'
    if raw.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    try:
        raw.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        return f'unknown (first 32 bytes: {raw[:32].hex()})'


def count_replacement_chars(text: str) -> int:
    """统计替换字符 (U+FFFD) 的出现次数。"""
    return text.count('\ufffd')


def safe_decode(raw: bytes) -> Tuple[str, str]:
    """安全解码，返回 (内容, 检测到的编码)。"""
    encoding = detect_encoding(raw)

    if encoding == 'utf-16-le':
        return raw.decode('utf-16-le'), encoding
    if encoding == 'utf-16-be':
        return raw.decode('utf-16-be'), encoding
    if encoding == 'utf-8-sig':
        return raw.decode('utf-8-sig'), encoding
    if encoding == 'utf-8':
        return raw.decode('utf-8'), encoding

    for enc in ['gbk', 'gb2312', 'big5', 'shift_jis', 'euc-jp', 'euc-kr']:
        try:
            return raw.decode(enc), enc
        except Exception:
            continue

    return raw.decode('utf-8', errors='replace'), 'utf-8(forced)'


def collect_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """收集需要处理的文件列表。"""
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
        text_exts = ['.md', '.txt', '.py', '.yaml', '.yml', '.json',
                     '.toml', '.cfg', '.ini', '.conf', '.css', '.html',
                     '.js', '.ts', '.xml', '.sh', '.ps1', '.bat']
        files = []
        for ext in text_exts:
            pattern = os.path.join(directory, '**', f'*{ext}')
            files.extend(glob.glob(pattern, recursive=True))
        return sorted(set(files))


def scan_directory(directory: str, extensions: Optional[List[str]] = None) -> None:
    """扫描目录并报告各文件的编码状态。"""
    files = collect_files(directory, extensions)
    if not files:
        sp(f"在 {directory} 中未找到匹配的文件")
        return

    non_utf8 = []
    clean_count = 0

    for filepath in files:
        if not os.path.isfile(filepath):
            continue

        rel = os.path.relpath(filepath, directory)
        try:
            with open(filepath, 'rb') as f:
                raw = f.read()

            encoding = detect_encoding(raw)
            if encoding == 'utf-8':
                clean_count += 1
            else:
                non_utf8.append((rel, encoding))
                sp(f"  [{encoding:>14}] {rel}")

        except Exception as e:
            sp(f"  [ERROR] {rel}: {e}")

    sp(f"\n总计: {clean_count} UTF-8, {len(non_utf8)} 非 UTF-8")


def preview_conversion(directory: str, extensions: Optional[List[str]] = None) -> None:
    """预览将进行的转换。"""
    files = collect_files(directory, extensions)
    if not files:
        sp(f"在 {directory} 中未找到匹配的文件")
        return

    to_convert = []
    for filepath in files:
        if not os.path.isfile(filepath):
            continue

        with open(filepath, 'rb') as f:
            raw = f.read()

        encoding = detect_encoding(raw)
        if encoding != 'utf-8':
            rel = os.path.relpath(filepath, directory)
            rep_count = count_replacement_chars(
                raw.decode('utf-8', errors='replace'))
            to_convert.append((rel, encoding, rep_count))

    if not to_convert:
        sp("所有文件已是 UTF-8，无需转换")
        return

    sp(f"以下 {len(to_convert)} 个文件将被转换为 UTF-8：")
    for rel, enc, rep_count in to_convert:
        rep_note = f" (含 {rep_count} 个替换字符)" if rep_count > 0 else ""
        sp(f"  {rel:<50} {enc:>12}{rep_note}")

    sp(f"\n总计: {len(to_convert)} 个文件待转换")


def convert_to_utf8(directory: str, extensions: Optional[List[str]] = None,
                    backup: bool = False) -> None:
    """将非 UTF-8 文件转换为 UTF-8。"""
    files = collect_files(directory, extensions)
    if not files:
        sp(f"在 {directory} 中未找到匹配的文件")
        return

    converted = 0
    skipped = 0
    errors = 0

    for filepath in files:
        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, 'rb') as f:
                raw = f.read()

            encoding = detect_encoding(raw)
            if encoding == 'utf-8':
                skipped += 1
                continue

            content, actual_enc = safe_decode(raw)

            if backup:
                bak_path = filepath + '.bak'
                shutil.copy2(filepath, bak_path)

            with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)

            rel = os.path.relpath(filepath, directory)
            rep_count = count_replacement_chars(content)
            rep_note = f" (含 {rep_count} 个替换字符)" if rep_count > 0 else ""
            sp(f"  {rel:<50} {actual_enc:>12} -> utf-8{rep_note}")
            converted += 1

        except Exception as e:
            rel = os.path.relpath(filepath, directory)
            sp(f"  ! {rel}: {e}")
            errors += 1

    sp(f"\n完成: {converted} 转换, {skipped} 跳过(已是UTF-8), {errors} 错误")


def check_replacement_chars(directory: str, extensions: Optional[List[str]] = None) -> None:
    """检查文件中的替换字符。"""
    files = collect_files(directory, extensions)
    if not files:
        sp(f"在 {directory} 中未找到匹配的文件")
        return

    found_any = False
    for filepath in files:
        if not os.path.isfile(filepath):
            continue

        with open(filepath, 'rb') as f:
            raw = f.read()

        content = raw.decode('utf-8', errors='replace')
        count = count_replacement_chars(content)

        if count > 0:
            found_any = True
            rel = os.path.relpath(filepath, directory)
            sp(f"  {rel:<60} {count} 个替换字符")

    if not found_any:
        sp("未发现替换字符")
    else:
        sp(f"\n建议运行: python scripts/fix_encoding.py convert <dir> --backup")


def main():
    parser = argparse.ArgumentParser(
        description="批量转换文件编码为 UTF-8"
    )
    parser.add_argument("mode",
                        choices=["scan", "preview", "convert", "check-replacement"],
                        help="操作模式")
    parser.add_argument("directory", nargs="?", default=".",
                        help="目标目录（默认当前目录）")
    parser.add_argument("--ext", default="",
                        help="文件扩展名过滤，逗号分隔，如 .md,.py,.yaml")
    parser.add_argument("--backup", action="store_true",
                        help="转换前创建 .bak 备份")

    args = parser.parse_args()
    extensions = [e.strip() for e in args.ext.split(",")
                  if e.strip()] if args.ext else None

    if args.mode == "scan":
        scan_directory(args.directory, extensions)
    elif args.mode == "preview":
        preview_conversion(args.directory, extensions)
    elif args.mode == "convert":
        convert_to_utf8(args.directory, extensions, args.backup)
    elif args.mode == "check-replacement":
        check_replacement_chars(args.directory, extensions)


if __name__ == '__main__':
    main()
