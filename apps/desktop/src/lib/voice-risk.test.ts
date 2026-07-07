import { describe, expect, it } from 'vitest'

import { isRiskyVoiceTranscript } from './voice-risk'

describe('isRiskyVoiceTranscript', () => {
  const risky = [
    // Destructive file operations
    'delete all the files in the downloads folder',
    'remove the old backup folder',
    'wipe everything in the temp directory',
    'run rm -rf on the build directory',
    'rm dash rf the build folder',
    'drop the users table from the database',
    'empty the trash',
    'format the drive',
    'overwrite the config file',
    '删除所有文件',
    // Git state changes
    'git push origin main',
    'commit the changes and push to main',
    'push the latest changes to origin',
    'force push the branch',
    'git commit everything with a message saying done',
    '重置 git 分支',
    '推送到 main 分支',
    // Publishing / deploying
    'deploy the site to production',
    'publish the package to npm',
    'publish it now',
    'ship the new version to the app store',
    '部署到生产环境',
    '发布这个版本',
    // Messages to third parties
    "send an email to the client saying we're done",
    'send a message to the team on slack',
    'text him that the demo is ready',
    '给客户发邮件说我们完成了',
    '发短信给他说明天见',
    // Purchases / payments
    'buy the domain for me',
    'purchase a license for the pro version',
    'pay the invoice',
    'pay the bill',
    'transfer the money to him',
    'wire the money to the vendor',
    'venmo him 50 dollars',
    'cancel the subscription',
    'unsubscribe from the paid plan',
    'subscribe to the pro plan',
    '购买这个域名',
    '支付账单',
    // Secrets / credentials
    'rotate the api keys',
    'change the database password',
    'delete the environment variables from the config',
    '更新 API 密钥',
    '删除环境变量'
  ]

  const safe = [
    '',
    '   ',
    "what's the weather like today",
    'summarize the last reply for me',
    'open the readme file and read it aloud',
    'the push notification design looks great',
    'explain how commits work in git',
    'tell me about the deployment process',
    'i bought a new keyboard yesterday',
    'did the message from alice mention the schedule',
    'what does this error message the app shows mean',
    'check out the PR and tell me what it does',
    'in order to fix it we should refactor the parser'
  ]

  it.each(risky)('flags risky transcript: %s', transcript => {
    expect(isRiskyVoiceTranscript(transcript)).toBe(true)
  })

  it('flags risky transcripts regardless of casing', () => {
    expect(isRiskyVoiceTranscript('Delete ALL the files in the Downloads folder')).toBe(true)
  })

  it.each(safe)('allows harmless transcript: %s', transcript => {
    expect(isRiskyVoiceTranscript(transcript)).toBe(false)
  })
})
