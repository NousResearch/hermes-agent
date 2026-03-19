"""
cosmos CI/CD Pipeline Management

"Good news, everyone! The build passed!"

Comprehensive CI/CD integration for GitHub Actions, GitLab CI, and Jenkins.
"""

from Cosmos.cicd.pipeline_manager import (
    PipelineManager,
    Pipeline,
    PipelineRun,
    PipelineStatus,
)
from Cosmos.cicd.github_actions import GitHubActionsManager
from Cosmos.cicd.gitlab_ci import GitLabCIManager
from Cosmos.cicd.jenkins_manager import JenkinsManager

__all__ = [
    "PipelineManager",
    "Pipeline",
    "PipelineRun",
    "PipelineStatus",
    "GitHubActionsManager",
    "GitLabCIManager",
    "JenkinsManager",
]
