"""MR 创建器 - 将代码修复封装为 Merge Request。"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from gitlab.gitlab_client import GitLabClient


class MRCreator:
    """MR 创建器。"""

    def __init__(self, config: Dict[str, Any]):
        self.client = GitLabClient(config.get("gitlab", {}))
        self.default_branch = config.get("gitlab", {}).get("default_branch", "main")

    def create_fix_mr(
        self,
        project_id: str,
        component: str,
        fault_type: str,
        file_path: str,
        original_content: str,
        fixed_content: str,
        diagnosis: str,
        branch_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建代码修复 MR。

        1. 创建修复分支
        2. 提交修复代码
        3. 创建 MR
        """
        if not self.client.is_configured():
            return {"success": False, "error": "GitLab 未配置"}

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if branch_name is None:
            branch_name = f"fix/{component}_{fault_type}_{now}"

        try:
            # 1. 创建分支
            self.client.create_branch(project_id, branch_name)

            # 2. 提交修复
            commit_msg = (
                f"fix: {component} {fault_type} 自动修复\n\n"
                f"- 故障类型: {fault_type}\n"
                f"- 诊断: {diagnosis}\n"
                f"- 由 MES AI 巡检系统自动生成"
            )
            self.client.create_or_update_file(
                project_id, file_path, fixed_content, commit_msg, branch=branch_name
            )

            # 3. 创建 MR
            mr_title = f"fix: {component} {fault_type} 自动修复 ({now})"
            mr_description = self._build_mr_description(
                component, fault_type, file_path, diagnosis, original_content, fixed_content
            )
            mr = self.client.create_merge_request(
                project_id, branch_name, mr_title, mr_description
            )

            return {
                "success": True,
                "mr_url": mr.get("web_url", ""),
                "mr_iid": mr.get("iid"),
                "branch": branch_name,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _build_mr_description(
        self,
        component: str,
        fault_type: str,
        file_path: str,
        diagnosis: str,
        original: str,
        fixed: str,
    ) -> str:
        """构建 MR 描述。"""
        return f"""## 故障背景

- **组件**: {component}
- **故障类型**: {fault_type}
- **文件**: `{file_path}`
- **时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## AI 诊断

{diagnosis}

## 修复方案

```diff
{self._generate_diff(original, fixed)}
```

## 测试

- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 人工 Review 通过

---
*由 MES AI 巡检系统自动生成，请 Review 后合并。*
"""

    def _generate_diff(self, original: str, fixed: str) -> str:
        """生成简单 diff。"""
        orig_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)

        diff = []
        max_lines = max(len(orig_lines), len(fixed_lines))
        for i in range(max_lines):
            orig = orig_lines[i].rstrip() if i < len(orig_lines) else ""
            fixed = fixed_lines[i].rstrip() if i < len(fixed_lines) else ""
            if orig != fixed:
                if orig:
                    diff.append(f"- {orig}")
                if fixed:
                    diff.append(f"+ {fixed}")

        return "\n".join(diff[:50])  # 最多 50 行 diff