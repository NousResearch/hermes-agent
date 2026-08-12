/**
 * Reference user widget: a live clock docked above the status bar.
 * Copy to ~/.hermes/tui-widgets/clock.mjs, then `/widgets-reload` and `/clock`.
 */
export default function register(sdk) {
  const { Box, Dialog, React, Text, defineWidgetApp, h } = sdk

  function Face({ label, t }) {
    const [now, setNow] = React.useState(() => new Date())

    React.useEffect(() => {
      const id = setInterval(() => setNow(new Date()), 1000)

      return () => clearInterval(id)
    }, [])

    return h(
      Box,
      { columnGap: 1, flexDirection: 'row' },
      h(Text, { bold: true, color: t.color.label }, label),
      h(Text, { color: t.color.text }, now.toLocaleTimeString('en-GB', { hour12: false, timeZone: label === 'local' ? undefined : label }))
    )
  }

  defineWidgetApp({
    id: 'clock',
    help: 'live clock in the dock (arg: IANA timezone)',
    mode: 'ambient',
    usage: 'usage: /clock [timezone]   e.g. /clock UTC · /clock Asia/Tokyo',

    init(arg) {
      const label = arg.trim() || 'local'

      try {
        new Date().toLocaleTimeString('en-GB', { timeZone: label === 'local' ? undefined : label })
      } catch {
        return null
      }

      return { label }
    },

    reduce(state, { ch, key }) {
      return key.escape || ch === 'q' ? null : state
    },

    render({ state, t }) {
      return h(Dialog, { width: 30 }, h(Face, { label: state.label, t }))
    }
  })
}
