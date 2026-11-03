/**
 * Strings for describe-a-bot drafting (New Agent "Describe the bot you want",
 * Edit Profile "Describe a change"). Kept beside the feature and plugged into
 * every Bot Mode locale under `draft`.
 */

// A type alias (not an interface) so it satisfies the plugin bundle's string index signature.
export type DraftMessages = {
  accept: string
  after: string
  appliesNext: string
  before: string
  changeLabel: string
  changePlaceholder: string
  describeLabel: string
  describePlaceholder: string
  done: string
  draft: string
  drafting: string
  dropped: (names: string) => string
  edit: string
  empty: string
  failed: (reason: string) => string
  keep: string
  noChanges: string
  noSuggestions: string
  notePlaceholder: string
  nothingSaved: string
  propose: string
  regenerate: string
  reject: string
  suggested: string
}

const en: DraftMessages = {
  accept: 'Accept',
  after: 'Proposed',
  appliesNext:
    'Accepted changes are written on Save and reach the bot when its chat next rebuilds its prompt (a new session or after compaction). The conversation in progress is not changed.',
  before: 'Now',
  changeLabel: 'Describe a change',
  changePlaceholder: 'e.g. Be less formal, and stay quiet when nothing changed',
  describeLabel: 'Describe the bot you want',
  describePlaceholder: 'e.g. Triages my inbox every morning, flags anything urgent, never replies for me',
  done: 'Done',
  draft: 'Draft',
  drafting: 'Drafting…',
  dropped: names => `Not installed on this machine, left out: ${names}`,
  edit: 'Edit',
  empty: '(empty)',
  failed: reason => `Couldn't draft: ${reason}`,
  keep: 'Keep',
  noChanges: 'No changes proposed.',
  noSuggestions: 'No capability suggestions.',
  notePlaceholder: 'Note for Regenerate (optional), e.g. "less formal"',
  nothingSaved: 'Nothing is saved until you click Create.',
  propose: 'Propose',
  regenerate: 'Regenerate',
  reject: 'Reject',
  suggested: 'Turned on at Create:'
}

const ja: DraftMessages = {
  accept: '採用',
  after: '提案',
  appliesNext:
    '採用した変更は「保存」で書き込まれ、ボットのチャットが次にプロンプトを組み直したとき（新しいセッションまたは圧縮後）に反映されます。進行中の会話は変更されません。',
  before: '現在',
  changeLabel: '変更内容を説明',
  changePlaceholder: '例: もう少しくだけた口調で。変化がないときは黙っていて',
  describeLabel: '作りたいボットを説明してください',
  describePlaceholder: '例: 毎朝受信箱を整理し、急ぎのものに印を付ける。代わりに返信はしない',
  done: '完了',
  draft: '下書き',
  drafting: '下書き中…',
  dropped: names => `このマシンに未インストールのため除外: ${names}`,
  edit: '編集',
  empty: '（空）',
  failed: reason => `下書きできませんでした: ${reason}`,
  keep: '維持',
  noChanges: '変更の提案はありません。',
  noSuggestions: '提案される機能はありません。',
  notePlaceholder: '再生成へのメモ（任意）例: 「もっとくだけて」',
  nothingSaved: '「作成」を押すまで何も保存されません。',
  propose: '提案',
  regenerate: '再生成',
  reject: '却下',
  suggested: '作成時にオンにする機能:'
}

const zh: DraftMessages = {
  accept: '采纳',
  after: '建议',
  appliesNext: '采纳的修改在“保存”时写入，并在该机器人的聊天下次重建提示词时（新会话或压缩后）生效。进行中的对话不会被改动。',
  before: '当前',
  changeLabel: '描述要做的修改',
  changePlaceholder: '例如：语气轻松一点，没有变化时保持安静',
  describeLabel: '描述你想要的机器人',
  describePlaceholder: '例如：每天早上整理我的收件箱，标出紧急邮件，从不替我回复',
  done: '完成',
  draft: '起草',
  drafting: '起草中…',
  dropped: names => `本机未安装，已略去：${names}`,
  edit: '编辑',
  empty: '（空）',
  failed: reason => `无法起草：${reason}`,
  keep: '保留',
  noChanges: '没有建议的修改。',
  noSuggestions: '没有建议的能力。',
  notePlaceholder: '重新生成的备注（可选），例如“更随意一些”',
  nothingSaved: '点击“创建”之前不会保存任何内容。',
  propose: '生成建议',
  regenerate: '重新生成',
  reject: '拒绝',
  suggested: '创建时开启：'
}

const zhHant: DraftMessages = {
  accept: '採用',
  after: '建議',
  appliesNext: '採用的修改在「儲存」時寫入，並在該機器人的聊天下次重建提示詞時（新工作階段或壓縮後）生效。進行中的對話不會被改動。',
  before: '目前',
  changeLabel: '描述要做的修改',
  changePlaceholder: '例如：語氣輕鬆一點，沒有變化時保持安靜',
  describeLabel: '描述你想要的機器人',
  describePlaceholder: '例如：每天早上整理我的收件匣，標出緊急郵件，從不替我回覆',
  done: '完成',
  draft: '起草',
  drafting: '起草中…',
  dropped: names => `本機未安裝，已略去：${names}`,
  edit: '編輯',
  empty: '（空）',
  failed: reason => `無法起草：${reason}`,
  keep: '保留',
  noChanges: '沒有建議的修改。',
  noSuggestions: '沒有建議的能力。',
  notePlaceholder: '重新產生的備註（可選），例如「更隨意一些」',
  nothingSaved: '點擊「建立」之前不會儲存任何內容。',
  propose: '產生建議',
  regenerate: '重新產生',
  reject: '拒絕',
  suggested: '建立時開啟：'
}

export const DRAFT_LOCALES = { en, ja, zh, zhHant }
