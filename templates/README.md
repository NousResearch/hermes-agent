# templates/（本 fork 附带模板）

本目录是 douzi457 fork 的附加模板，不影响 Hermes 运行时，只供部署时取用。

| 文件 | 用途 |
|---|---|
| `soul/SOUL.zh-CN.md` | 本 fork 的出厂默认人格+纪律（中文版），与 `hermes_cli/default_soul.py` 的 `DEFAULT_SOUL_MD` 内容一致。想改默认人格改代码常量，这个文件只是可读副本与备份。 |
| `soul/SOUL.en.md` | 上游英文版默认 SOUL（Nous Research 原文）。想切回英文默认：把它的内容粘贴到你的 `~/.hermes/SOUL.md` 即可（实例级覆盖，不用改代码）。 |
| `AGENTS.zh-CN.md` | 项目级纪律模板。复制到某个项目根目录改名为 `AGENTS.md`，Hermes 在该项目工作时自动加载，是全局铁律在单项目里的落地版。 |

说明：出厂默认（代码常量）只放一份中文版；英文版以模板文件附带，是本 fork 的既定选择（2026-10）。
