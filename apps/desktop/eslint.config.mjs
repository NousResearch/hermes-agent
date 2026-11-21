import shared from '../../eslint.config.shared.mjs'
import globals from 'globals'

// Drift guard G2 (DESIGN.md "Portal design language"): redesign-only TSX takes its look from the
// tokens in src/app/redesign/tokens/, not from arbitrary Tailwind values. Flags, inside className /
// cn() strings: an arbitrary text, leading, tracking, radius, colour, ring, shadow, fill or stroke value
// unless it is a bare token read (a var() of a token), Tailwind's own text sizes and scaled radii (the
// redesign's --radius-scalar makes those land between Figma's steps), and hex colours. Arbitrary layout
// values (grid templates, min() widths) are fine.
const VISUAL_ARBITRARY =
  /^-?(text|leading|tracking|rounded(-[a-z]{1,2})?|bg|border(-[a-z]{1,2})?|ring|shadow|fill|stroke|outline|decoration)-\[(.+)\]$/
const TOKEN_READ = /^(?:[a-z-]+:)?var\(--[\w-]+(?:,[^\]]*)?\)$/
const SCALED = /^(rounded(-[a-z]{1,2})?-(xs|sm|md|lg|xl|2xl|3xl)|text-(xs|sm|base|lg))$/
const CLASS_CALLS = new Set(['cn', 'clsx', 'cva', 'twMerge'])

const utilityOf = token => {
  let depth = 0
  let cut = 0

  for (let i = 0; i < token.length; i++) {
    const ch = token[i]
    depth += ch === '[' ? 1 : ch === ']' ? -1 : 0

    if (ch === ':' && depth === 0) {
      cut = i + 1
    }
  }

  return token.slice(cut).replace(/^!/, '')
}

const offendingClass = token => {
  const utility = utilityOf(token)
  const arbitrary = VISUAL_ARBITRARY.exec(utility)

  if (arbitrary) {
    return !TOKEN_READ.test(arbitrary[4])
  }

  return SCALED.test(utility) || /#[0-9a-f]{3,8}\b/i.test(utility)
}

const inClassContext = ancestors =>
  ancestors.some(
    node =>
      (node.type === 'JSXAttribute' && ['className', 'class'].includes(node.name.name)) ||
      (node.type === 'CallExpression' && CLASS_CALLS.has(node.callee.name))
  )

const noPortalArbitrary = {
  meta: {
    type: 'suggestion',
    messages: {
      arbitrary:
        '`{{cls}}` restates a design value in redesign-only markup. Use a token-backed class or a CSS rule that reads src/app/redesign/tokens/ (DESIGN.md "Portal design language").'
    },
    schema: []
  },
  create(context) {
    const check = (node, text) => {
      if (!inClassContext(context.sourceCode.getAncestors(node))) {
        return
      }

      for (const cls of text.split(/\s+/).filter(Boolean)) {
        if (offendingClass(cls)) {
          context.report({ node, messageId: 'arbitrary', data: { cls } })
        }
      }
    }

    return {
      Literal: node => typeof node.value === 'string' && check(node, node.value),
      TemplateElement: node => check(node, node.value.cooked ?? '')
    }
  }
}

export default [
  ...shared,
  {
    // Desktop is an Electron renderer — it legitimately uses browser globals
    // (window, document, etc). Re-add them here; the shared config omits
    // globals.browser so terminal-only workspaces (ui-tui) don't get them.
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node
      }
    }
  },
  {
    // THE PLUGIN FENCE: plugins speak @hermes/plugin-sdk (+ react), never `@/…`
    // internals — the same isolation a runtime-fetched published plugin gets,
    // enforced on bundled ones so the SDK surface stays honest and sufficient.
    files: ['src/plugins/**/*.{ts,tsx}'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['@/*', '../*', '@hermes/shared'],
              message: 'Plugins import only @hermes/plugin-sdk (and react). Missing something? Add it to the SDK.'
            }
          ]
        }
      ]
    }
  },
  {
    // Redesign-only markup (mixed files with a classic branch are not judged yet). Warn while the
    // existing sites burn down (DESIGN.md "To unify in Phase B"); then it becomes an error.
    files: [
      'src/app/redesign/**/*.tsx',
      'src/app/chat/sidebar/portal/**/*.tsx',
      'src/app/agent-settings/**/*.tsx',
      'src/components/portal/**/*.tsx',
      'src/app/settings/settings-card.tsx',
      'src/app/settings/appearance-overview.tsx'
    ],
    ignores: ['src/**/*.test.tsx', 'src/app/agent-settings/classic-settings.tsx'],
    plugins: { 'hermes-desktop': { rules: { 'no-portal-arbitrary': noPortalArbitrary } } },
    rules: { 'hermes-desktop/no-portal-arbitrary': 'warn' }
  },
  {
    files: ['**/*.test.tsx'],
    rules: {
      'no-restricted-globals': ['warn', 'document']
    }
  },
  {
    // crypto.randomUUID is secure-context-only: a Webapp opened over plain HTTP from another
    // machine has none, so every id minted with it threw and nothing was sent.
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/**/*.test.{ts,tsx}'],
    rules: {
      'no-restricted-properties': [
        'error',
        {
          property: 'randomUUID',
          message: 'Use randomUuid from @hermes/shared (plugins: @hermes/plugin-sdk); it works on insecure origins.'
        }
      ]
    }
  },
  {
    // Ban mirroring reactive values into refs via useEffect — the "atom-mirrored
    // ref" antipattern. A ref synced from a nanostores atom via useEffect lags the
    // atom by one render, which creates stale-read bugs in callbacks that read the
    // ref (cancelRun sent session.interrupt to the wrong session; steerPrompt,
    // restoreToMessage, editMessage all had closure-priority stale reads). The fix
    // is to read $atom.get() directly in callbacks instead. This rule catches the
    // mirroring effect at lint time so the pattern can't reappear. Legitimate
    // non-atom ref writes inside useEffect (DOM instance refs, mount flags, request
    // tokens, prop mirrors) get an eslint-disable-next-line with a comment.
    files: ['src/**/*.{ts,tsx}'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          // useEffect(() => { someRef.current = value }, [value])
          selector:
            'CallExpression[callee.name="useEffect"] > ArrowFunctionExpression[body.type="AssignmentExpression"][body.left.type="MemberExpression"][body.left.property.name="current"]',
          message:
            'Do not mirror reactive values into refs via useEffect. Read $atom.get() directly in callbacks instead — refs synced from atoms lag one render and cause stale-read bugs.'
        },
        {
          // useEffect(() => { someRef.current = value; ... }, [value])
          selector:
            'CallExpression[callee.name="useEffect"] > ArrowFunctionExpression[body.type="BlockStatement"]:has(AssignmentExpression[left.type="MemberExpression"][left.property.name="current"])',
          message:
            'Do not mirror reactive values into refs via useEffect. Read $atom.get() directly in callbacks instead — refs synced from atoms lag one render and cause stale-read bugs.'
        },
        {
          // useEffect(() => { setMutableRef(ref, value) }, [value])
          selector:
            'CallExpression[callee.name="useEffect"] > ArrowFunctionExpression[body.type="BlockStatement"]:has(CallExpression[callee.name="setMutableRef"])',
          message:
            'Do not mirror reactive values into refs via useEffect (setMutableRef included). Read $atom.get() directly in callbacks instead — refs synced from atoms lag one render and cause stale-read bugs.'
        },
        {
          // {contribution.render()} anywhere inside JSX — calling a render
          // callback inline makes its hooks belong to the HOST component, so
          // loading/replacing a plugin changes the host's hook count → React
          // #310 (#80560, crashed every Windows user with a desktop plugin).
          // Mount it as a child instead: <ContribRender render={c.render} />.
          selector: 'JSXExpressionContainer CallExpression[callee.property.name="render"]',
          message:
            'Do not call render() callbacks inline in JSX — the callback\u2019s hooks become the host\u2019s and plugin load/replace changes the host hook count (React #310). Mount it as a component: <ContribRender render={...} /> from @/contrib/react/boundary.'
        }
      ]
    }
  }
]
