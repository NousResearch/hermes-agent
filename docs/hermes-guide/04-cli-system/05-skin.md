# 主题皮肤系统

## 4.1 概述

**核心文件**: `hermes_cli/skin_engine.py` (~35,000 字符)

**设计**: 数据驱动的 YAML 皮肤系统，无需修改代码即可定制 UI 外观。

---

## 4.2 皮肤结构

```yaml
# ~/.hermes/skins/my-skin.yaml
name: my-skin
version: 1.0

colors:
  primary: "#6366f1"      # 主色 (indigo)
  secondary: "#8b5cf6"    # 次色 (violet)
  background: "#0f172a"   # 深色背景
  foreground: "#f1f5f9"   # 浅色文字
  accent: "#22d3ee"       # 强调色 (cyan)
  error: "#ef4444"        # 错误色
  success: "#22c55e"      # 成功色

fonts:
  mono: "JetBrains Mono"
  sans: "Inter"

terminal:
  theme: "catppuccin-mocha"
```

---

## 4.3 内置皮肤

| 皮肤名 | 说明 |
|--------|------|
| `default` | 默认主题 |
| `ares` | 暗色主题 |
| `mono` | 纯单色主题 |
| `slate` | 石板灰主题 |

---

## 4.4 皮肤引擎

```python
# hermes_cli/skin_engine.py
class SkinEngine:
    """皮肤引擎"""

    def __init__(self, skin_name: str = "default"):
        self.skin = self.load_skin(skin_name)

    def load_skin(self, name: str) -> dict:
        """加载皮肤"""
        # 1. 先查找用户皮肤
        user_skin = Path(f"~/.hermes/skins/{name}.yaml")
        if user_skin.exists():
            return yaml.safe_load(user_skin.read_text())

        # 2. 查找内置皮肤
        built_in = BUILT_IN_SKINS.get(name)
        if built_in:
            return built_in

        # 3. 默认皮肤
        return BUILT_IN_SKINS["default"]

    def apply(self, component: str) -> str:
        """应用皮肤到组件"""
        return self.skin.get(component, {})
```

---

## 4.5 关键源码索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| 皮肤引擎 | `hermes_cli/skin_engine.py` | `SkinEngine` |
| 内置皮肤 | `hermes_cli/skin_engine.py` | `BUILT_IN_SKINS` |
| 皮肤加载 | `hermes_cli/skin_engine.py` | `load_skin()` |

---

## 4.6 下一步

- [Gateway 系统](../../05-gateway/01-overview.md) - 网关架构总览
- [会话管理](../../05-gateway/02-session.md) - session.py 详解
- [平台适配器](../../05-gateway/03-platforms.md) - 平台实现

---

[← 返回总目录](../SUMMARY.md) | [← 认证](04-auth.md) | [Gateway 架构 →](../../05-gateway/01-overview.md)
