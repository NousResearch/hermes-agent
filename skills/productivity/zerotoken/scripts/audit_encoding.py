"""
audit_encoding.py — 全项目编码合规审计（对照 Unicode 安全编码规范）

检查项：
1. 每个文本文件的编码（UTF-8 无 BOM / UTF-8 带 BOM / UTF-16 / 其他）
2. 替换字符 U+FFFD 计数（编码损坏痕迹）
3. 混合换行符（LF/CRLF）统计
4. 二进制文件识别（跳过，不检查内容）

输出：UTF-8 文件，避免终端 GBK 乱码。
用法：
    python scripts/audit_encoding.py [--root .] [--out audit_result.txt]
"""

import argparse
import os
from typing import Dict, List, Tuple

TEXT_EXTS = {
    '.md', '.txt', '.py', '.yaml', '.yml', '.json', '.toml',
    '.cfg', '.ini', '.conf', '.css', '.html', '.js', '.ts',
    '.xml', '.sh', '.ps1', '.bat', '.cmd', '.csv', '.log',
    '.gitignore', '.gitattributes', '.env', '.editorconfig',
}

SKIP_DIRS = {'.git', '.codegraph', '.reasonix', '__pycache__', 'node_modules', '.venv', 'venv'}
BINARY_EXTS = {'.png', '.jpg', '.jpeg', '.ico', '.pdf', '.db', '.gif', '.bmp'}


def is_binary(raw: bytes) -> bool:
    """识别二进制文件。

    先按 NUL 字节判断；再尝试 UTF-8 整体解码（中文多字节序列
    在此视为文本而非"不可打印"）；两者都不过时检查控制字符比例。
    注意：不能用"采样窗口内可打印字节比例"判断——中文 UTF-8 文件
    前几 KB 可能全是多字节字符，会被误判为二进制。
    """
    if b'\x00' in raw:
        return True
    try:
        raw.decode('utf-8')
        return False
    except UnicodeDecodeError:
        pass
    # 非 UTF-8：控制字符（CR/LF/TAB 除外）占比高视为二进制
    sample = raw[:4096]
    if not sample:
        return False
    control = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
    return control / len(sample) > 0.3


def detect(raw: bytes) -> str:
    if raw.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
        return 'utf-16'
    try:
        raw.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        return 'non-utf8'


def line_ending_stats(text: str) -> Tuple[int, int]:
    crlf = text.count('\r\n')
    lf = text.count('\n') - crlf
    return lf, crlf


def collect_files(root: str) -> List[str]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            files.append(os.path.join(dirpath, name))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', default='audit_result.txt')
    args = parser.parse_args()

    lines: List[str] = []
    lines.append('=' * 70)
    lines.append('编码合规审计 (Unicode 安全编码规范对照)')
    lines.append(f'扫描根目录: {os.path.abspath(args.root)}')
    lines.append('=' * 70)

    stats: Dict[str, int] = {}
    problems: List[str] = []
    binary_count = 0

    for fp in collect_files(args.root):
        rel = os.path.relpath(fp, args.root)
        ext = os.path.splitext(fp)[1].lower()

        try:
            with open(fp, 'rb') as f:
                raw = f.read()
        except OSError as e:
            problems.append(f'[读失败] {rel}: {e}')
            continue

        if is_binary(raw) or ext in BINARY_EXTS:
            binary_count += 1
            continue

        enc = detect(raw)
        stats[enc] = stats.get(enc, 0) + 1

        if enc == 'non-utf8':
            problems.append(f'[非UTF-8] {rel}')
            continue

        text = raw.decode('utf-8-sig' if enc == 'utf-8-sig' else 'utf-8')
        rep = text.count('\ufffd')
        if rep > 0:
            problems.append(f'[替换字符x{rep}] {rel}')

        lf, crlf = line_ending_stats(text)
        if crlf > 0 and lf > 0:
            problems.append(f'[混合换行 LF={lf} CRLF={crlf}] {rel}')

    lines.append(f'文本文件: {sum(stats.values())}  二进制/跳过: {binary_count}')
    for enc, n in sorted(stats.items()):
        lines.append(f'  {enc:<12} {n} 个文件')

    lines.append('')
    if problems:
        lines.append(f'发现问题 {len(problems)} 项:')
        for p in problems:
            lines.append(f'  {p}')
    else:
        lines.append('未发现问题。所有文本文件均为 UTF-8，无替换字符，无混合换行。')

    content = '\n'.join(lines) + '\n'
    out_path = args.out
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)

    # 用 write_result 同款方式提示绝对路径（避免终端打印中文）
    abs_path = os.path.abspath(out_path)
    print(f'audit written to: {abs_path}')


if __name__ == '__main__':
    main()
