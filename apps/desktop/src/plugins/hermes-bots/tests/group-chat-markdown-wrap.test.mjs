import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadPreRenderer(isLikelyStructuredText) {
  const start = pluginSource.indexOf('const Streamdown =')
  const end = pluginSource.indexOf('\nconst ID =', start)

  assert.ok(start >= 0 && end > start, 'optional SDK capability section exists')

  const context = {
    cn: (...values) => values.filter(Boolean).join(' '),
    jsx: (type, props) => ({ props, type }),
    sdk: {
      Streamdown: () => null,
      ...(isLikelyStructuredText ? { isLikelyStructuredText } : {})
    }
  }
  const source = `${pluginSource.slice(start, end)}\nglobalThis.__markdown = { GroupChatPre, GROUP_CHAT_MARKDOWN_COMPONENTS };\n`

  vm.runInNewContext(source, context, { filename: 'plugin.js' })

  return context.__markdown
}

function code(language, body) {
  return { props: { children: body, className: `language-${language}` }, type: 'code' }
}

test('plain text prose wraps while preserving the pre element contract', () => {
  const calls = []
  const markdown = loadPreRenderer(body => {
    calls.push(body)
    return false
  })
  const body =
    'Please review this ordinary group-chat update. It contains several natural-language sentences with spaces and should remain readable when the chat becomes narrow.'
  const children = code('text', body)
  const rendered = markdown.GroupChatPre({
    children,
    className: 'existing',
    node: { internal: true },
    title: 'message'
  })

  assert.equal(rendered.type, 'pre')
  assert.equal(rendered.props.children, children)
  assert.equal(rendered.props.title, 'message')
  assert.equal(Object.hasOwn(rendered.props, 'node'), false)
  assert.match(rendered.props.className, /whitespace-pre-wrap/)
  assert.match(rendered.props.className, /wrap-anywhere/)
  assert.doesNotMatch(rendered.props.className, /overflow-x-auto/)
  assert.deepEqual(calls, [body])
  assert.equal(markdown.GROUP_CHAT_MARKDOWN_COMPONENTS.pre, markdown.GroupChatPre)
})

test('structured text keeps horizontal scroll', () => {
  const body =
    'Host production\n  HostName example.internal\n  IdentityFile ~/.ssh/id_ed25519\n  StrictHostKeyChecking yes'
  const markdown = loadPreRenderer(value => value === body)
  const rendered = markdown.GroupChatPre({ children: code('text', body) })

  assert.match(rendered.props.className, /overflow-x-auto/)
  assert.doesNotMatch(rendered.props.className, /whitespace-pre-wrap|wrap-anywhere/)
})

test('other languages and unexpected Streamdown trees fail closed to scroll', () => {
  let calls = 0
  const markdown = loadPreRenderer(() => {
    calls += 1
    return false
  })
  const javascript = markdown.GroupChatPre({ children: code('javascript', 'const value = 1') })
  const nonCodeElement = markdown.GroupChatPre({
    children: { props: { children: 'ordinary prose', className: 'language-text' }, type: 'span' }
  })
  const unexpected = markdown.GroupChatPre({ children: 'plain child' })

  assert.match(javascript.props.className, /overflow-x-auto/)
  assert.match(nonCodeElement.props.className, /overflow-x-auto/)
  assert.match(unexpected.props.className, /overflow-x-auto/)
  assert.equal(calls, 0)
})

test('legacy SDK without the classifier loads and keeps text scrolling', () => {
  const markdown = loadPreRenderer(undefined)
  const rendered = markdown.GroupChatPre({ children: code('text', 'Ordinary prose from an older Desktop build.') })

  assert.match(rendered.props.className, /overflow-x-auto/)
  assert.doesNotMatch(rendered.props.className, /whitespace-pre-wrap|wrap-anywhere/)
})
