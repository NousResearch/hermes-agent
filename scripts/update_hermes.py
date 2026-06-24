"""Hermes Agent 自动更新脚本

功能：
1. 备份当前本地修改（使用 git stash）
2. 拉取最新源码
3. 恢复本地修改
4. 更新依赖
5. 验证运行

用法：
    python scripts/update_hermes.py [--dry-run] [--force]
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERMES_DIR = Path(r"D:\hermes-agent")
BACKUP_DIR = HERMES_DIR / ".update_backups"
LOG_FILE = HERMES_DIR / ".update_log"


def run_cmd(cmd, cwd=None, check=True):
    """运行命令并返回结果"""
    cmd_str = " ".join(str(c) for c in cmd)
    print("  $ " + cmd_str)
    result = subprocess.run(
        cmd,
        cwd=cwd or HERMES_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    if check and result.returncode != 0:
        print("  X 失败: " + result.stderr[:500])
        return None
    return result


def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = "[" + timestamp + "] " + message
    print(log_line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def backup_local_changes():
    """备份本地修改（使用 git stash）"""
    log("=== 备份本地修改 ===")
    
    # 创建备份目录
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / backup_time
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # 获取 modified 文件列表
    result = run_cmd(["git", "status", "--short"], check=False)
    if not result or not result.stdout.strip():
        log("  OK 没有本地修改需要备份")
        return backup_path, []
    
    # 解析 modified 和 untracked 文件
    modified_files = []
    untracked_files = []
    
    for line in result.stdout.strip().split("\n"):
        if line.startswith(" M ") or line.startswith("M "):
            modified_files.append(line[3:].strip())
        elif line.startswith("?? "):
            untracked_files.append(line[3:].strip())
    
    log("  发现 " + str(len(modified_files)) + " 个 modified 文件, " + str(len(untracked_files)) + " 个 untracked 文件")
    
    # 保存 git diff patch
    if modified_files:
        diff_result = run_cmd(["git", "diff"], check=False)
        if diff_result and diff_result.stdout:
            diff_file = backup_path / "local_changes.patch"
            diff_file.write_text(diff_result.stdout, encoding="utf-8")
            log("  OK 修改已保存到: " + str(diff_file))
    
    # 保存 untracked 文件列表（不复制内容，避免递归问题）
    if untracked_files:
        untracked_list = backup_path / "untracked_files.txt"
        untracked_list.write_text("\n".join(untracked_files), encoding="utf-8")
        log("  OK untracked 文件列表已保存到: " + str(untracked_list))
        log("  注意: untracked 文件未复制内容，请手动备份重要文件")
    
    # 保存文件列表
    manifest = backup_path / "manifest.txt"
    manifest_content = "Backup: " + backup_time + "\n"
    manifest_content += "Modified files:\n"
    if modified_files:
        manifest_content += "\n".join(modified_files) + "\n"
    else:
        manifest_content += "None\n"
    manifest_content += "\nUntracked files:\n"
    if untracked_files:
        manifest_content += "\n".join(untracked_files) + "\n"
    else:
        manifest_content += "None\n"
    manifest.write_text(manifest_content, encoding="utf-8")
    
    return backup_path, modified_files + untracked_files


def stash_changes():
    """Stash 本地修改"""
    log("=== Stash 本地修改 ===")
    
    # 检查是否有 stash
    result = run_cmd(["git", "stash", "list"], check=False)
    if result and result.stdout.strip():
        log("  现有 stash: " + str(len(result.stdout.strip().split("\n"))) + " 个")
    
    # stash 所有修改（包括 untracked）
    stash_msg = "auto-backup-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    result = run_cmd(["git", "stash", "push", "-u", "-m", stash_msg], check=False)
    if result and result.returncode == 0:
        log("  OK 修改已 stash")
        return True
    else:
        log("  ! stash 失败或没有修改")
        return False


def pull_latest():
    """拉取最新代码"""
    log("=== 拉取最新代码 ===")
    
    # 获取当前分支
    result = run_cmd(["git", "branch", "--show-current"], check=False)
    if not result:
        log("  X 无法获取当前分支")
        return False
    
    branch = result.stdout.strip()
    log("  当前分支: " + branch)
    
    # 获取远程更新
    result = run_cmd(["git", "fetch", "origin"], check=False)
    if not result:
        log("  X fetch 失败")
        return False
    log("  OK fetch 完成")
    
    # 检查落后多少 commit
    result = run_cmd(["git", "rev-list", "HEAD..origin/" + branch, "--count"], check=False)
    if result:
        behind_count = result.stdout.strip()
        log("  落后 origin/" + branch + ": " + behind_count + " commits")
    
    # 执行 pull
    result = run_cmd(["git", "pull", "origin", branch], check=False)
    if not result or result.returncode != 0:
        log("  X pull 失败")
        if result and result.stderr:
            log("  错误: " + result.stderr[:500])
        return False
    
    log("  OK pull 完成")
    return True


def restore_changes(backup_path, stashed):
    """恢复本地修改"""
    log("=== 恢复本地修改 ===")
    
    # 尝试 stash pop
    if stashed:
        result = run_cmd(["git", "stash", "pop"], check=False)
        if result and result.returncode == 0:
            log("  OK stash 已恢复")
            return True
        else:
            log("  ! stash pop 失败，可能需要手动解决冲突")
            log("  使用: git stash pop")
            return False
    
    # 如果没有 stash，尝试应用 patch
    patch_file = backup_path / "local_changes.patch"
    if patch_file.exists():
        result = run_cmd(["git", "apply", str(patch_file)], check=False)
        if result and result.returncode == 0:
            log("  OK patch 已应用")
            return True
        else:
            log("  ! patch 应用失败，可能需要手动解决")
            log("  手动应用: git apply " + str(patch_file))
            return False
    
    log("  没有需要恢复的修改")
    return True


def update_dependencies():
    """更新依赖"""
    log("=== 更新依赖 ===")
    
    # 检查是否有 requirements.txt 或 pyproject.toml
    req_file = HERMES_DIR / "requirements.txt"
    pyproject = HERMES_DIR / "pyproject.toml"
    
    if pyproject.exists():
        log("  使用 pyproject.toml")
        # 尝试 pip install -e .
        result = run_cmd([sys.executable, "-m", "pip", "install", "-e", "."], check=False)
        if result and result.returncode == 0:
            log("  OK 依赖更新完成")
            return True
    elif req_file.exists():
        log("  使用 requirements.txt")
        result = run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=False)
        if result and result.returncode == 0:
            log("  OK 依赖更新完成")
            return True
    
    log("  ! 依赖更新可能需要手动处理")
    return False


def verify_installation():
    """验证安装"""
    log("=== 验证安装 ===")
    
    # 检查 hermes 命令
    result = run_cmd([sys.executable, "-m", "hermes", "--version"], check=False)
    if result and result.returncode == 0:
        log("  OK hermes 版本: " + result.stdout.strip())
        return True
    
    # 尝试直接导入
    result = run_cmd([sys.executable, "-c", "import hermes_agent; print('OK')"], check=False)
    if result and result.returncode == 0:
        log("  OK hermes_agent 模块可导入")
        return True
    
    log("  ! 验证失败，可能需要手动检查")
    return False


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent 自动更新")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不执行实际修改")
    parser.add_argument("--force", action="store_true", help="强制更新，不提示确认")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖更新")
    parser.add_argument("--skip-verify", action="store_true", help="跳过验证")
    args = parser.parse_args()
    
    log("=" * 60)
    log("Hermes Agent 自动更新")
    log("=" * 60)
    
    if not HERMES_DIR.exists():
        log("X Hermes 目录不存在: " + str(HERMES_DIR))
        sys.exit(1)
    
    # 确认
    if not args.force and not args.dry_run:
        response = input("确认更新 Hermes Agent? [y/N]: ")
        if response.lower() != "y":
            log("已取消")
            sys.exit(0)
    
    # 备份
    backup_path, changed_files = backup_local_changes()
    
    if args.dry_run:
        log("=== 模拟模式，停止 ===")
        log("将备份到: " + str(backup_path))
        log("将更新 " + str(len(changed_files)) + " 个文件")
        sys.exit(0)
    
    # Stash
    stashed = stash_changes()
    
    # Pull
    if not pull_latest():
        log("X 更新失败")
        sys.exit(1)
    
    # 恢复修改
    restore_changes(backup_path, stashed)
    
    # 更新依赖
    if not args.skip_deps:
        update_dependencies()
    
    # 验证
    if not args.skip_verify:
        verify_installation()
    
    log("=" * 60)
    log("更新完成")
    log("备份位置: " + str(backup_path))
    log("=" * 60)


if __name__ == "__main__":
    main()
