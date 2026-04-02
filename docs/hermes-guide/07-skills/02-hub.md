# Skills Hub 集成

## 7.1 概述

**核心文件**: `tools/skills_hub.py` (~98,000 字符)

**功能**: 与 agentskills.io 社区技能市场集成，支持技能的发现、安装和分享。

---

## 7.2 Hub API

```python
# tools/skills_hub.py
class SkillsHubClient:
    """Skills Hub API 客户端"""

    BASE_URL = "https://agentskills.io/api"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SKILLS_HUB_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def list_skills(self, category: str = None) -> List[dict]:
        """列出技能"""
        params = {"category": category} if category else {}
        response = self.session.get(f"{self.BASE_URL}/skills", params=params)
        return response.json()["skills"]

    def get_skill(self, skill_id: str) -> dict:
        """获取技能详情"""
        response = self.session.get(f"{self.BASE_URL}/skills/{skill_id}")
        return response.json()

    def install_skill(self, skill_id: str) -> None:
        """安装技能到本地"""
        skill = self.get_skill(skill_id)

        # 创建目录
        skill_dir = Path.home() / ".hermes" / "skills" / skill["name"]
        skill_dir.mkdir(parents=True, exist_ok=True)

        # 下载文件
        for file in skill["files"]:
            content = self.session.get(file["url"]).content
            (skill_dir / file["name"]).write_bytes(content)
```

---

## 7.3 关键源码索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| Hub 客户端 | `tools/skills_hub.py` | `SkillsHubClient` |
| 技能安装 | `tools/skills_hub.py` | `install_skill()` |
| 技能列表 | `tools/skills_hub.py` | `list_skills()` |

---

## 7.4 下一步

- [ACP 适配器](../../08-advanced/01-acp.md) - 编辑器集成
- [Cron 调度](../../08-advanced/02-cron.md) - 定时任务

---

[← 返回总目录](../SUMMARY.md) | [← Skills 概述](01-overview.md) | [ACP →](../../08-advanced/01-acp.md)
