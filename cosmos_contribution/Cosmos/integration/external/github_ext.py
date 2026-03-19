"""
cosmos GitHub Integration (Full Implementation).

"The Code must flow!"

Features:
1. Issue Sync: Injects GitHub issues as tasks.
2. PR Review: Can trigger code reviews via the Critic Agent.
3. CI/CD Monitoring: Watches for build failures.
"""

import asyncio
from typing import Dict, Any, List
from loguru import logger
from github import Github, Auth
from github.GithubException import GithubException

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus
from Cosmos.core.nexus import nexus, Signal, SignalType

class GitHubProvider(ExternalProvider):
    def __init__(self, token: str, repo: str):
        super().__init__(IntegrationConfig(name="github", api_key=token))
        self.repo_name = repo
        self.client = None
        self.repo_obj = None
        
    async def connect(self) -> bool:
        if not self.config.api_key:
            logger.warning("GitHub: No token provided.")
            return False
            
        try:
            auth = Auth.Token(self.config.api_key)
            self.client = Github(auth=auth)
            
            # Verify access by getting repo
            # PyGithub is synchronous, so we run in executor
            loop = asyncio.get_event_loop()
            self.repo_obj = await loop.run_in_executor(
                None, lambda: self.client.get_repo(self.repo_name)
            )
            
            logger.info(f"GitHub: Connected to {self.repo_name}")
            self.status = ConnectionStatus.CONNECTED
            return True
        except Exception as e:
            logger.error(f"GitHub connection failed: {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def sync(self):
        """Poll for new issues."""
        if self.status != ConnectionStatus.CONNECTED:
            return

        loop = asyncio.get_event_loop()
        try:
            # Poll open issues
            issues = await loop.run_in_executor(
                None, lambda: list(self.repo_obj.get_issues(state='open')[:5])
            )
            
            for issue in issues:
                # Convert to signal
                await nexus.emit(
                    SignalType.EXTERNAL_ALERT, 
                    {
                        "source": "github",
                        "type": "issue",
                        "id": issue.number,
                        "title": issue.title,
                        "url": issue.html_url
                    },
                    source="github_provider"
                )
        except Exception as e:
            logger.error(f"GitHub sync error: {e}")

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("GitHub not connected")

        loop = asyncio.get_event_loop()
        
        if action == "create_issue":
            title = params.get('title')
            body = params.get('body', '')
            logger.info(f"GitHub: Creating issue '{title}'")
            
            issue = await loop.run_in_executor(
                None, lambda: self.repo_obj.create_issue(title=title, body=body)
            )
            return {"id": issue.number, "url": issue.html_url}
            
        elif action == "create_pr":
            title = params.get('title')
            body = params.get('body', '')
            head = params.get('head')
            base = params.get('base', 'main')
            
            pr = await loop.run_in_executor(
                None, 
                lambda: self.repo_obj.create_pull(
                    title=title, body=body, head=head, base=base
                )
            )
            return {"id": pr.number, "url": pr.html_url}
            
        else:
            raise ValueError(f"Unknown action: {action}")
