"""MES AI 巡检系统安装脚本。"""

import json
import os
import shutil
import sys
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录。"""
    return Path(__file__).parent.parent


def setup_hermes_home():
    """设置 MES_INSPECTION_HOME 环境变量。"""
    project_root = get_project_root()
    home_dir = project_root / "data"
    home_dir.mkdir(exist_ok=True)
    print(f"MES_INSPECTION_HOME: {home_dir}")
    return home_dir


def install_skills():
    """安装 Skills 到 ~/.hermes/skills/。"""
    project_root = get_project_root()
    skills_src = project_root / "skills"
    skills_dst = Path.home() / ".hermes" / "skills" / "mes-inspection"

    if skills_dst.exists():
        print(f"Skills 目录已存在: {skills_dst}")
        resp = input("是否覆盖? [y/N] ").strip().lower()
        if resp != "y":
            print("跳过 Skills 安装")
            return

    skills_dst.mkdir(parents=True, exist_ok=True)
    for skill_dir in skills_src.iterdir():
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
            dst = skills_dst / skill_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(skill_dir, dst)
            print(f"  ✅ 安装 Skill: {skill_dir.name}")

    print(f"Skills 已安装到: {skills_dst}")


def install_cron_jobs(dry_run: bool = False):
    """安装 Cron 任务。"""
    project_root = get_project_root()
    cron_config = project_root / "config" / "cron_jobs.json"

    if not cron_config.exists():
        print("❌ cron_jobs.json 不存在")
        return

    with open(cron_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"\n📋 将创建 {len(config['jobs'])} 个 Cron 任务:")
    for job in config["jobs"]:
        print(f"  - {job['name']}: {job['schedule']}")

    if dry_run:
        print("\n[DRY RUN] 未实际创建任务")
        return

    print("\n要创建 Cron 任务，请在 Hermes Agent 中执行：")
    print("  hermes gateway start")
    print("  然后通过 /cron 命令或 cronjob 工具创建任务")
    print("\n或者使用以下命令：")
    for job in config["jobs"]:
        skills_str = json.dumps(job.get("skills", []))
        toolsets_str = json.dumps(job.get("enabled_toolsets", ["terminal"]))
        print(f'\n  cronjob create \\')
        print(f'    --name "{job["name"]}" \\')
        print(f'    --schedule "{job["schedule"]}" \\')
        print(f'    --prompt "{job["prompt"][:50]}..." \\')
        print(f'    --skills \'{skills_str}\' \\')
        print(f'    --deliver feishu')


def create_config():
    """创建默认配置文件。"""
    project_root = get_project_root()
    config_dst = Path.home() / ".mes-inspection" / "config.yaml"
    config_example = project_root / "config" / "mes_inspection.yaml.example"

    if config_dst.exists():
        print(f"配置文件已存在: {config_dst}")
        return

    config_dst.parent.mkdir(parents=True, exist_ok=True)
    if config_example.exists():
        shutil.copy2(config_example, config_dst)
        print(f"✅ 配置文件已创建: {config_dst}")
        print("   请编辑配置文件，填入实际的连接信息和阈值")
    else:
        print("⚠️ 配置示例文件不存在")


def main():
    """主安装流程。"""
    print("=" * 50)
    print("MES AI 巡检系统 - 安装向导")
    print("=" * 50)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "skills":
            install_skills()
        elif cmd == "cron":
            dry_run = "--dry-run" in sys.argv
            install_cron_jobs(dry_run=dry_run)
        elif cmd == "config":
            create_config()
        elif cmd == "home":
            setup_hermes_home()
        else:
            print(f"未知命令: {cmd}")
            print("用法: python setup/install.py [skills|cron|config|home]")
    else:
        # 完整安装
        setup_hermes_home()
        create_config()
        install_skills()
        install_cron_jobs(dry_run=True)
        print("\n✅ 安装完成！")
        print("\n下一步：")
        print("  1. 编辑 ~/.mes-inspection/config.yaml")
        print("  2. 设置环境变量: export MES_INSPECTION_HOME=<项目路径>")
        print("  3. 启动 Hermes Gateway: hermes gateway start")
        print("  4. 创建 Cron 任务: python setup/install.py cron")


if __name__ == "__main__":
    main()
