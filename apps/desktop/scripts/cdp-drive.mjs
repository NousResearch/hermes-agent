#!/usr/bin/env node
// cdp-drive.mjs — reusable CDP driver for the Hermes desktop app (dev mode).
//
// Attaches to the running `npm run dev` Electron renderer over CDP (port 9222,
// opened by electron/main.cjs in dev) and exposes the recurring verification
// moves as composable subcommands, so L3 checks don't need a fresh throwaway
// script every time:
//
//   node scripts/cdp-drive.mjs frames                       # list frames + body preview
//   node scripts/cdp-drive.mjs snapshot [path]              # screenshot (default /tmp/cdp-snap.png)
//   node scripts/cdp-drive.mjs prompt "文本"                 # type into composer + Enter
//   node scripts/cdp-drive.mjs wait-card [uri-substr]       # poll for a card iframe (default any ui://)
//   node scripts/cdp-drive.mjs card-text [uri-substr]       # innerText of newest matching card frame
//   node scripts/cdp-drive.mjs card-wait-text <regex> [uri] # poll card text until regex matches
//   node scripts/cdp-drive.mjs drive "文本" <regex> [png]    # prompt + wait for card text match + snapshot
//   node scripts/cdp-drive.mjs audit                        # card matrix sweep -> .tmp/card-audit.md
//   node scripts/cdp-drive.mjs record "文本" [secs]          # prompt + capture gateway event stream -> .tmp/event-log.json
//
// Prompt-submitting commands open a FRESH session first (so runs never pollute
// the session a human is using); set CDP_SAME_SESSION=1 to opt out.
// Renderer console errors are collected and echoed as [console-error].
//
// Options (env): CDP_URL (default http://127.0.0.1:9222), CDP_TIMEOUT_S (default 120).
// Bridge frames ([mcp-app...]) from the renderer console are always echoed.
import { mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

// Sandboxed shells may deny mkdtemp in the system TMPDIR (EPERM on
// /var/folders/...). Point playwright at a workspace-local tmp dir before it
// loads; must happen before the dynamic import below.
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const tmpDir = join(repoRoot, '.tmp')

mkdirSync(tmpDir, { recursive: true })
process.env.TMPDIR = process.env.TMPDIR?.startsWith(repoRoot) ? process.env.TMPDIR : tmpDir

// Via @playwright/test, not `playwright` — the former is this workspace's own
// declared devDependency, so the driver needs no extra root-level dep.
const { chromium } = await import('@playwright/test')

const CDP_URL = process.env.CDP_URL || 'http://127.0.0.1:9222'
const TIMEOUT_S = Number(process.env.CDP_TIMEOUT_S || 120)
const sleep = ms => new Promise(r => setTimeout(r, ms))

async function attach() {
    const browser = await chromium.connectOverCDP(CDP_URL)
    const ctx = browser.contexts()[0]
    const page = ctx.pages().find(p => p.url().includes('127.0.0.1:5174')) ?? ctx.pages()[0]

    if (!page) {
        throw new Error('no renderer page found — is `npm run dev` running?')
    }

    page.on('console', m => {
        const t = m.text()

        if (t.includes('mcp-app')) {
            console.log('[bridge]', t.slice(0, 200))
        }

        if (m.type() === 'error') {
            consoleErrors.push(t.slice(0, 300))
            console.log('[console-error]', t.slice(0, 200))
        }
    })
    page.on('pageerror', e => {
        consoleErrors.push(String(e).slice(0, 300))
        console.log('[pageerror]', String(e).slice(0, 200))
    })

    return { browser, page }
}

const consoleErrors = []

async function newSession(page) {
    await page.locator('text=New session').first().click()
    await page.waitForSelector('[contenteditable="true"]', { timeout: 10000 })
}

async function listCardIframes(page) {
    return page.evaluate(() => {
        return [...document.querySelectorAll('iframe')].map(f => ({
            title: f.title,
            srcdocLen: (f.getAttribute('srcdoc') || '').length
        }))
    })
}

async function submitPrompt(page, text) {
    // Fresh session by default so scripted runs never pollute a session the
    // human is working in (learned the hard way). CDP_SAME_SESSION=1 opts out.
    if (!process.env.CDP_SAME_SESSION) {
        await newSession(page)
    }

    let composer = page.locator('[contenteditable="true"]').first()

    // Self-heal: the app may be parked on a route without a composer
    // (e.g. #/messaging). Jump into a fresh chat session first.
    if (!(await composer.count())) {
        console.log('[prompt] no composer on current route — opening a new session')
        await newSession(page)
        composer = page.locator('[contenteditable="true"]').first()
    }

    await composer.click()
    await page.keyboard.press('Meta+A')
    await page.keyboard.press('Backspace')
    await composer.type(text)
    await page.keyboard.press('Enter')
    console.log('[prompt] submitted:', text.slice(0, 80))
}

async function newestCardText(page) {
    // Newest tool card = LAST iframe in DOM order (transcript order). frames()
    // order is attach order and lies when older sessions left cards around.
    const count = await page.locator('iframe').count()

    if (!count) {
        return ''
    }

    try {
        return await page
            .frameLocator(`iframe >> nth=${count - 1}`)
            .locator('body')
            .innerText({ timeout: 3000 })
    } catch {
        return ''
    }
}

async function waitCard(page, uriSubstr) {
    for (let i = 0; i < TIMEOUT_S / 2; i++) {
        const cards = (await listCardIframes(page)).filter(c => c.title.includes(uriSubstr ?? 'ui://'))

        if (cards.length) {
            console.log('[wait-card] found:', JSON.stringify(cards))

            return true
        }

        await sleep(2000)
    }

    console.log('[wait-card] TIMEOUT')

    return false
}

async function waitCardText(page, regex) {
    const re = new RegExp(regex)

    for (let i = 0; i < TIMEOUT_S / 2; i++) {
        const txt = await newestCardText(page)

        if (re.test(txt)) {
            console.log('[card-text]', txt.replace(/\n+/g, ' | ').slice(0, 400))
            console.log('[wait-text] MATCHED', regex)

            return true
        }

        await sleep(2000)
    }

    console.log('[wait-text] TIMEOUT for', regex)

    return false
}

async function snapshot(page, path) {
    const out = path || '/tmp/cdp-snap.png'

    await page.screenshot({ path: out })
    console.log('[snapshot]', out)
}

// Card matrix: one scenario per utp UI surface. Each runs in a fresh session.
// `expect` is matched against the newest card's innerText; failures and
// console errors land in the audit report for human review.
const AUDIT_SCENARIOS = [
    {
        name: 'catalog-search',
        prompt: '用 mcp_utp_utp_catalog_search 搜索 keyword=蓝牙耳机 search_type=KEYWORD_SEARCH limit=5，不要问我问题',
        expect: '¥|好评|售'
    },
    {
        name: 'catalog-product',
        prompt: '用 mcp_utp_utp_catalog_search 搜蓝牙耳机(KEYWORD_SEARCH, limit=3)，然后用 mcp_utp_utp_catalog_product 打开第一个商品详情，不要问我问题',
        expect: '起批|库存|规格|颜色'
    },
    {
        name: 'cart-after-add',
        prompt: '用 mcp_utp_utp_catalog_search 搜蓝牙耳机(KEYWORD_SEARCH, limit=3)，把第一个商品用 mcp_utp_utp_cart_add 加购2件，然后用 mcp_utp_utp_cart_list 展示购物车，不要问我问题',
        expect: '购物车|合计|数量|结算'
    },
    {
        name: 'order-list',
        prompt: '用 mcp_utp_utp_order_list 展示我的订单列表，不要问我问题',
        expect: '订单|暂无|登录'
    },
    {
        name: 'address-form',
        prompt: '用 mcp_utp_utp_address_form 打开收货地址表单，不要问我问题',
        expect: '地址|收货|登录'
    },
    {
        name: 'login-card',
        prompt: '调用 mcp_utp_utp_login 打开登录卡片，不要问我问题',
        expect: '登录|扫码|账号'
    }
]

async function audit(page) {
  const { writeFileSync, mkdirSync: mkdir } = await import('node:fs')
    const outDir = join(tmpDir, 'card-audit')

    mkdir(outDir, { recursive: true })
    const rows = []

    for (const sc of AUDIT_SCENARIOS) {
        console.log(`\n===== audit: ${sc.name} =====`)
        const errBefore = consoleErrors.length

        await submitPrompt(page, sc.prompt)
        const matched = await waitCardText(page, sc.expect)
        const text = (await newestCardText(page)).replace(/\n+/g, ' | ')
        const shot = join(outDir, `${sc.name}.png`)

        await snapshot(page, shot)
        rows.push({
            name: sc.name,
            matched,
            errors: consoleErrors.slice(errBefore),
            text: text.slice(0, 300),
            shot
        })
    }

    const report = [
        '# MCP Apps card audit',
        `run: ${new Date().toISOString()}`,
        '',
        '| scenario | text-match | console errors |',
        '|---|---|---|',
        ...rows.map(r => `| ${r.name} | ${r.matched ? '✅' : '❌'} | ${r.errors.length} |`),
        '',
        ...rows.flatMap(r => [
            `## ${r.name}`,
            `- match: ${r.matched}`,
            `- shot: ${r.shot}`,
            r.errors.length ? `- errors:\n${r.errors.map(e => `  - ${e}`).join('\n')}` : '- errors: none',
            `- card text: ${r.text}`,
            ''
        ])
    ].join('\n')
    const reportPath = join(tmpDir, 'card-audit.md')

    writeFileSync(reportPath, report)
    console.log('\n[audit] report:', reportPath)

    return rows.every(r => r.matched)
}

const [cmd, ...args] = process.argv.slice(2)
const { browser, page } = await attach()
let exitCode = 0

try {
    if (cmd === 'frames') {
        console.log('[page]', page.url())

        for (const f of page.frames()) {
            let body = ''

            try {
                body = await f.evaluate(() => (document.body?.innerText || '').slice(0, 80))
            } catch (e) {
                body = 'EVAL FAIL: ' + String(e).slice(0, 60)
            }

            console.log('-', JSON.stringify(f.url()).slice(0, 50), '::', body.replace(/\n/g, ' '))
        }

        console.log('[iframes]', JSON.stringify(await listCardIframes(page)))
    } else if (cmd === 'snapshot') {
        await snapshot(page, args[0])
    } else if (cmd === 'prompt') {
        await submitPrompt(page, args[0] ?? '')
    } else if (cmd === 'wait-card') {
        exitCode = (await waitCard(page, args[0])) ? 0 : 1
    } else if (cmd === 'card-text') {
        console.log((await newestCardText(page)).replace(/\n+/g, ' | ').slice(0, 600))
    } else if (cmd === 'card-wait-text') {
        exitCode = (await waitCardText(page, args[0])) ? 0 : 1
    } else if (cmd === 'drive') {
        await submitPrompt(page, args[0] ?? '')
        exitCode = (await waitCardText(page, args[1] ?? '.')) ? 0 : 1
        await snapshot(page, args[2] && !/^ui:/.test(args[2]) ? args[2] : '/tmp/cdp-drive.png')
    } else if (cmd === 'audit') {
        exitCode = (await audit(page)) ? 0 : 1
    } else if (cmd === 'record') {
        // Capture the real gateway event stream (via the DEV window.__hermesEventLog
        // tap) while a prompt runs — evidence for transcript-lifecycle bugs.
        const { writeFileSync } = await import('node:fs')
        const secs = Number(args[1] || 90)
        const events = []
        let lastSeq = 0

        await submitPrompt(page, args[0] ?? '')

        for (let i = 0; i < secs / 2; i++) {
            await sleep(2000)

            const batch = await page.evaluate(after => window.__hermesEventLog?.drain(after) ?? [], lastSeq)

            for (const ev of batch) {
                events.push(ev)
                lastSeq = Math.max(lastSeq, ev.seq)
                console.log(`[ev ${ev.seq}] ${ev.type}`, JSON.stringify(ev.payload ?? {}).slice(0, 150))
            }

            // Stop early once the turn completed and things went quiet.
            if (events.some(e => e.type === 'message.complete') && batch.length === 0) {
                break
            }
        }

        const outPath = join(tmpDir, 'event-log.json')

        writeFileSync(outPath, JSON.stringify(events, null, 1))
        console.log(`[record] ${events.length} events -> ${outPath}`)
        exitCode = events.length ? 0 : 1
    } else {
        console.log('usage: cdp-drive.mjs <frames|snapshot|prompt|wait-card|card-text|card-wait-text|drive|audit|record> [args]')
        exitCode = 2
    }
} finally {
    await browser.close()
}

process.exit(exitCode)
