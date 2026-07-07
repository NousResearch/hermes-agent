/**
 * Best-effort heuristic for hands-free voice transcripts. The voice
 * conversation loop auto-submits whatever it hears, so obvious commands that
 * could destroy files, rewrite git state, publish, spend money, message third
 * parties, or touch secrets are held for manual confirmation instead.
 *
 * This is not a complete policy engine. Keep it biased toward holding clear
 * high-risk intents, but expect the backend approval system to remain the real
 * enforcement layer for dangerous tool execution.
 */

// Patterns run against lowercased, whitespace-collapsed text. Most English
// patterns pair a risky verb with a nearby target so harmless mentions ("the
// push notification design", "the deployment process") stay below the bar.
const RISKY_PATTERNS: RegExp[] = [
  // Shell-destructive removals, including how STT tends to render `rm -rf`.
  /\brm\s+(?:-|dash |minus )/,
  /\brimraf\b/,
  // Deleting or overwriting files, folders, repos, branches, databases, config.
  /\b(?:delete|remove|erase|wipe|destroy|drop|purge|trash|empty|format|overwrite)\b.{0,60}\b(?:files?|folders?|director(?:y|ies)|drives?|trash|configs?|config files?|repositor(?:y|ies)|repos?|branch(?:es)?|databases?|tables?|backups?|everything|node modules)\b/,
  // Git commands that rewrite history or remote state.
  /\bgit\s+(?:commit|push|reset|clean|rebase|merge|revert)\b/,
  /\b(?:commit|push)\b.{0,50}\b(?:changes?|code|work|files?|branch(?:es)?|repo|origin|remote|main|master|everything)\b/,
  /\bforce[\s-]?push/,
  // Publishing / deploying / releasing.
  /\b(?:deploy|publish|release|ship)\b.{0,60}\b(?:apps?|sites?|websites?|packages?|builds?|versions?|updates?|changes?|prod(?:uction)?|live|npm|store|staging)\b/,
  /\b(?:deploy|publish|release|ship)\s+(?:it|this|that|now|everything)\b/,
  /\bgo\s+live\b/,
  // Outbound messages to third parties. Avoid "error message" noun phrases.
  /\b(?:send|forward|shoot)\b.{0,50}\b(?:e-?mails?|messages?|dms?|texts?|sms|slack|tweets?|replies|invoices?)\b/,
  /\b(?:email|text|dm|ping|slack)\s+(?:him|her|them|everyone|the\s+\w+|[\w.]+@)/,
  /\bmessage\s+(?:him|her|them|everyone|[\w.]+@)/,
  // Spending money and subscription changes. Carve out everyday phrases.
  /\b(?:buy|purchase|pay(?!\s+attention)|transfer|wire|venmo|checkout)\b.{0,60}\b(?:licen[cs]es?|domains?|plans?|subscriptions?|credits?|servers?|invoices?|bills?|money|dollars?|vendors?|versions?|it|this|that|one)\b/,
  /\b(?<!in\s)order\b.{0,60}\b(?:licen[cs]es?|domains?|plans?|subscriptions?|credits?|servers?|invoices?|bills?|it|this|that|one)\b/,
  /\bplace\s+(?:an?\s+)?order\b/,
  /\b(?:subscribe\s+to|unsubscribe\s+from|cancel\s+(?:the\s+)?subscription)\b/,
  // Secrets / credentials / environment configuration.
  /\b(?:rotate|revoke|regenerate|delete|remove|change|update|set|edit)\b.{0,50}\b(?:secrets?|api ?keys?|tokens?|credentials?|passwords?|environment variables?|env (?:vars?|variables?|files?)|\.env)\b/,
  // Chinese high-risk transcripts (zh locale support; broad by design).
  /(?:删除|移除|清空|擦除|格式化|覆盖).{0,20}(?:文件|文件夹|目录|硬盘|驱动器|垃圾箱|数据库|表|备份|环境变量|配置|仓库|分支|所有)/,
  /(?:git\s*)?(?:提交|推送|重置|强推).{0,20}(?:git|代码|更改|分支|main|master|远程|仓库)?/,
  /(?:部署|发布|上线|发版).{0,20}(?:生产|线上|网站|应用|版本|npm|商店|这个|现在)?/,
  /(?:发邮件|发送邮件|发短信|发消息|私信|通知).{0,20}(?:客户|团队|他|她|他们|所有人|用户)/,
  /(?:客户|团队|他|她|他们|所有人|用户).{0,20}(?:发邮件|发送邮件|发短信|发消息|私信|通知)/,
  /(?:购买|买|支付|付款|转账|订阅|取消订阅).{0,20}(?:域名|账单|发票|套餐|订阅|许可证|积分|服务器|钱|款)/,
  /(?:更新|删除|移除|轮换|吊销|重置|修改|设置).{0,20}(?:api\s*密钥|密钥|令牌|凭证|密码|环境变量|\.env)/
]

export function isRiskyVoiceTranscript(text: string): boolean {
  const normalized = text.toLowerCase().replace(/\s+/g, ' ').trim()

  if (!normalized) {
    return false
  }

  return RISKY_PATTERNS.some(pattern => pattern.test(normalized))
}
