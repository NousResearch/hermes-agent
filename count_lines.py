import os
from collections import defaultdict

# Mapping from file extension to language name
EXTENSION_MAP = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.ts': 'TypeScript',
    '.jsx': 'JavaScript (React)',
    '.tsx': 'TypeScript (React)',
    '.html': 'HTML',
    '.htm': 'HTML',
    '.css': 'CSS',
    '.scss': 'SCSS',
    '.sass': 'Sass',
    '.less': 'Less',
    '.md': 'Markdown',
    '.txt': 'Text',
    '.json': 'JSON',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.xml': 'XML',
    '.sql': 'SQL',
    '.java': 'Java',
    '.c': 'C',
    '.h': 'C/C++ Header',
    '.cpp': 'C++',
    '.hpp': 'C++ Header',
    '.cs': 'C#',
    '.php': 'PHP',
    '.rb': 'Ruby',
    '.go': 'Go',
    '.rs': 'Rust',
    '.swift': 'Swift',
    '.kt': 'Kotlin',
    '.kts': 'Kotlin Script',
    '.scala': 'Scala',
    '.sh': 'Shell',
    '.bash': 'Bash',
    '.zsh': 'Zsh',
    '.fish': 'Fish',
    '.ps1': 'PowerShell',
    '.bat': 'Batch',
    '.cmd': 'Batch',
    '.dockerfile': 'Dockerfile',
    '.gitignore': 'GitIgnore',
    '.gitattributes': 'GitAttributes',
    '.editorconfig': 'EditorConfig',
    '.ini': 'INI',
    '.cfg': 'Config',
    '.conf': 'Config',
    '.toml': 'TOML',
    '.xml': 'XML',
    '.xsl': 'XSL',
    '.xslt': 'XSLT',
    '.svg': 'SVG',
    '.png': 'PNG Image',
    '.jpg': 'JPEG Image',
    '.jpeg': 'JPEG Image',
    '.gif': 'GIF Image',
    '.bmp': 'BMP Image',
    '.tiff': 'TIFF Image',
    '.ico': 'ICO Image',
    '.webp': 'WebP Image',
    '.mp3': 'MP3 Audio',
    '.wav': 'WAV Audio',
    '.mp4': 'MP4 Video',
    '.avi': 'AVI Video',
    '.mov': 'MOV Video',
    '.pdf': 'PDF Document',
    '.zip': 'ZIP Archive',
    '.tar': 'TAR Archive',
    '.gz': 'GZIP Archive',
    '.bz2': 'BZIP2 Archive',
    '.xz': 'XZ Archive',
    '.7z': '7Z Archive',
    '.rar': 'RAR Archive',
    # Add more as needed
}

def get_language(file_path):
    """Get language based on file extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    return EXTENSION_MAP.get(ext, 'Other')

def count_lines(file_path):
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return len(f.readlines())
    except Exception:
        return 0

def main():
    # The repository root is the current directory (~/hermes-source)
    repo_root = os.getcwd()
    # We'll skip certain directories that are not part of the source code
    skip_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', 'dist', 'build', '.idea', '.vscode'}
    
    # Statistics
    lang_stats = defaultdict(lambda: {'files': 0, 'lines': 0})
    
    for root, dirs, files in os.walk(repo_root):
        # Skip directories in skip_dirs
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            # Get relative path for display? Not needed for counting.
            lang = get_language(file_path)
            lines = count_lines(file_path)
            lang_stats[lang]['files'] += 1
            lang_stats[lang]['lines'] += lines
    
    # Calculate total lines
    total_lines = sum(stats['lines'] for stats in lang_stats.values())
    
    # Prepare rows for the table
    rows = []
    for lang, stats in sorted(lang_stats.items(), key=lambda x: x[1]['lines'], reverse=True):
        files_count = stats['files']
        lines_count = stats['lines']
        percentage = (lines_count / total_lines * 100) if total_lines > 0 else 0
        rows.append((lang, files_count, lines_count, f"{percentage:.2f}%"))
    
    # Add total row
    rows.append(('TOTAL', sum(stats['files'] for stats in lang_stats.values()), total_lines, '100.00%'))
    
    # Generate markdown table
    header = "| 语言 | 文件数 | 总行数 | 占比 |"
    separator = "|------|--------|--------|------|"
    table_lines = [header, separator]
    for lang, files_count, lines_count, percentage in rows:
        table_lines.append(f"| {lang} | {files_count} | {lines_count} | {percentage} |")
    
    markdown_table = '\n'.join(table_lines)
    
    # Output to the target file
    target_dir = os.path.expanduser('~/Documents/HermesVault/02_知识')
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, '2026-06-22_codex测试_代码行数统计.md')
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    
    print(f"Markdown table written to {target_file}")

if __name__ == '__main__':
    main()
