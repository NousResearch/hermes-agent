# PDR 用户端（智能问答）

用户端只有问答相关页面（登录、问答、发现、收藏、历史），与管理端是两个独立工程，页面之间没有任何互相跳转的入口。

## 运行

推荐从仓库根目录一键启动：

```bash
python start_run.py
```

只看用户端界面时，也可以单独运行：

```bash
cd pdr/web/portal
pnpm install
pnpm dev
```

打开 http://127.0.0.1:5174 。

## 示例数据模式

后端问答接口完成前，用户端默认使用内置的示例数据（`src/api/mock.ts`），登录、流式回答、出处预览、追问、收藏、反馈、历史会话都能完整体验。示例数据保存在浏览器本地，清除站点数据即可还原。

- 演示模式下任意账号密码都能登录。
- 能命中示例资料的问题：儿童门票、外滩店退改 / 早餐、春季线路出发日期、导游投诉处理、老人参团；其他问题会如实提示“没有找到相关内容”。

切换到真实接口：把 `pdr/config.yaml` 的 `portal_web.mock_api` 改为 `false`（单独运行时设置环境变量 `VITE_PORTAL_MOCK=false`），请求会发往 `/api/portal/v1`，开发服务器把它代理到 `app.host:app.port`。

## 接口约定

全部类型在 `src/api/types.ts`，示例实现与真实实现共用同一个 `PortalApi` 接口。流式回答为 SSE，事件顺序：

```
retrieval → citations → delta… → follow_ups → done
```

## 目录

```
src/
├── api/          types.ts 接口约定 · http.ts 真实接口 · mock.ts 示例数据
├── stores/       auth（登录态）· chat（会话、提问、停止、反馈、收藏）
├── components/   侧边栏、资料范围、输入框、回答、出处抽屉、光球、图标
├── views/        登录、问答（首页 + 会话）、发现、收藏、历史
├── lib/          回答文本解析（段落 / 列表 / 加粗 / [n] 角标）、问候语、时间分组、图标路径
└── styles/       全局变量与通用样式
```
