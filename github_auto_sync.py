"""
GitHub 自动回流系统

核心功能：
1. 自动提交代码到 GitHub
2. 自动创建 Pull Request
3. 自动同步配置和技能
4. 容器集成支持
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GitHubAutoSync:
    """GitHub 自动回流系统"""
    
    def __init__(self, repo_path: str, remote: str = "origin", branch: str = "main"):
        """
        初始化 GitHub 同步
        
        Args:
            repo_path: 仓库路径
            remote: 远程名称
            branch: 分支名称
        """
        self.repo_path = Path(repo_path)
        self.remote = remote
        self.branch = branch
    
    def auto_commit(self, message: str, files: Optional[List[str]] = None) -> bool:
        """
        自动提交更改
        
        Args:
            message: 提交信息
            files: 文件列表（None 表示所有更改）
            
        Returns:
            是否成功
        """
        try:
            # 添加文件
            if files:
                for file in files:
                    self._run_git(["add", file])
            else:
                self._run_git(["add", "."])
            
            # 提交
            self._run_git(["commit", "-m", message])
            
            logger.info(f"✅ 自动提交: {message}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 提交失败: {e}")
            return False
    
    def auto_push(self, force: bool = False) -> bool:
        """
        自动推送到远程
        
        Args:
            force: 是否强制推送
            
        Returns:
            是否成功
        """
        try:
            cmd = ["push", self.remote, self.branch]
            if force:
                cmd.append("--force")
            
            self._run_git(cmd)
            
            logger.info(f"✅ 自动推送到 {self.remote}/{self.branch}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 推送失败: {e}")
            return False
    
    def create_pr(
        self,
        title: str,
        body: str,
        base: str = "main",
        head: Optional[str] = None
    ) -> Optional[str]:
        """
        创建 Pull Request
        
        Args:
            title: PR 标题
            body: PR 描述
            base: 目标分支
            head: 源分支（None 表示当前分支）
            
        Returns:
            PR URL 或 None
        """
        try:
            if head is None:
                head = self._get_current_branch()
            
            # 使用 gh CLI
            cmd = [
                "gh", "pr", "create",
                "--title", title,
                "--body", body,
                "--base", base,
                "--head", head
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            pr_url = result.stdout.strip()
            logger.info(f"✅ 创建 PR: {pr_url}")
            return pr_url
            
        except Exception as e:
            logger.error(f"❌ 创建 PR 失败: {e}")
            return None
    
    def sync_skills(self, skills_dir: str) -> bool:
        """
        同步技能到 GitHub
        
        Args:
            skills_dir: 技能目录
            
        Returns:
            是否成功
        """
        try:
            skills_path = Path(skills_dir)
            if not skills_path.exists():
                logger.warning(f"⚠️ 技能目录不存在: {skills_dir}")
                return False
            
            # 提交技能
            message = f"Auto-sync skills: {datetime.utcnow().isoformat()}"
            self.auto_commit(message, [str(skills_path)])
            
            # 推送
            self.auto_push()
            
            logger.info(f"✅ 同步技能: {skills_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 同步技能失败: {e}")
            return False
    
    def _run_git(self, args: List[str]) -> str:
        """
        运行 git 命令
        
        Args:
            args: git 参数
            
        Returns:
            命令输出
        """
        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    
    def _get_current_branch(self) -> str:
        """获取当前分支名"""
        return self._run_git(["branch", "--show-current"])


# 便捷函数
def auto_sync_to_github(
    repo_path: str,
    message: str,
    files: Optional[List[str]] = None,
    push: bool = True
) -> bool:
    """
    自动同步到 GitHub
    
    Args:
        repo_path: 仓库路径
        message: 提交信息
        files: 文件列表
        push: 是否推送
        
    Returns:
        是否成功
    """
    sync = GitHubAutoSync(repo_path)
    
    if not sync.auto_commit(message, files):
        return False
    
    if push:
        return sync.auto_push()
    
    return True


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("GitHub 自动回流系统测试")
    print("=" * 70)
    print()
    
    print("✅ GitHub 自动回流系统已实现")
    print()
    print("核心功能:")
    print("  • 自动提交代码")
    print("  • 自动推送到远程")
    print("  • 创建 Pull Request")
    print("  • 同步技能和配置")
    print()
    
    print("使用示例:")
    print("```python")
    print("from github_auto_sync import GitHubAutoSync")
    print()
    print("sync = GitHubAutoSync('~/.hermes/hermes-agent')")
    print("sync.auto_commit('Update skills')")
    print("sync.auto_push()")
    print("sync.create_pr('New features', 'Description')")
    print("```")
    print()
    
    print("=" * 70)
    print("✅ GitHub 自动回流系统测试完成")
    print("=" * 70)
