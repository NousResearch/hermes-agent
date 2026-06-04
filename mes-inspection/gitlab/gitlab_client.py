"""GitLab API 客户端。"""

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


class GitLabClient:
    """GitLab REST API 客户端。

    使用 urllib 实现，无额外依赖。
    """

    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get("url", "").rstrip("/")
        self.token = os.getenv(config.get("token_env", "GITLAB_TOKEN"), "")
        self.default_branch = config.get("default_branch", "main")
        self.reviewers = config.get("mr_reviewers", [])

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """发送 GitLab API 请求。"""
        url = f"{self.base_url}/api/v4{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"

        headers = {
            "PRIVATE-TOKEN": self.token,
            "Content-Type": "application/json",
        }

        body = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitLab API 错误 {e.code}: {error_body}")

    def get_project(self, project_id: str) -> Dict[str, Any]:
        """获取项目信息。"""
        return self._request("GET", f"/projects/{self._encode_id(project_id)}")

    def create_branch(
        self, project_id: str, branch_name: str, ref: str = None
    ) -> Dict[str, Any]:
        """创建分支。"""
        if ref is None:
            ref = self.default_branch
        return self._request("POST", f"/projects/{self._encode_id(project_id)}/repository/branches", {
            "branch": branch_name,
            "ref": ref,
        })

    def get_file(
        self, project_id: str, file_path: str, ref: str = None
    ) -> Dict[str, Any]:
        """获取文件内容。"""
        if ref is None:
            ref = self.default_branch
        encoded_path = urllib.request.quote(file_path, safe="")
        return self._request(
            "GET",
            f"/projects/{self._encode_id(project_id)}/repository/files/{encoded_path}",
            params={"ref": ref},
        )

    def create_or_update_file(
        self,
        project_id: str,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str = None,
        action: str = "auto",
    ) -> Dict[str, Any]:
        """创建或更新文件。

        action: "create" | "update" | "auto"（自动判断）
        """
        if branch is None:
            branch = self.default_branch

        if action == "auto":
            try:
                self.get_file(project_id, file_path, ref=branch)
                action = "update"
            except RuntimeError:
                action = "create"

        encoded_path = urllib.request.quote(file_path, safe="")
        return self._request(
            "POST",
            f"/projects/{self._encode_id(project_id)}/repository/commits",
            {
                "branch": branch,
                "commit_message": commit_message,
                "actions": [
                    {
                        "action": action,
                        "file_path": file_path,
                        "content": content,
                    }
                ],
            },
        )

    def create_merge_request(
        self,
        project_id: str,
        source_branch: str,
        title: str,
        description: str = "",
        target_branch: str = None,
        reviewer_ids: List[int] = None,
    ) -> Dict[str, Any]:
        """创建 Merge Request。"""
        if target_branch is None:
            target_branch = self.default_branch

        data = {
            "source_branch": source_branch,
            "target_branch": target_branch,
            "title": title,
            "description": description,
        }
        if reviewer_ids:
            data["reviewer_ids"] = reviewer_ids
        elif self.reviewers:
            data["reviewer_ids"] = self.reviewers

        return self._request(
            "POST",
            f"/projects/{self._encode_id(project_id)}/merge_requests",
            data,
        )

    def _encode_id(self, project_id: str) -> str:
        """编码项目 ID（支持数字 ID 和 URL 编码的路径）。"""
        if project_id.isdigit():
            return project_id
        return urllib.request.quote(project_id, safe="")

    def is_configured(self) -> bool:
        """检查是否已配置。"""
        return bool(self.base_url and self.token)