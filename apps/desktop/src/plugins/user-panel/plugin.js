import {
  Button,
  cn,
  Codicon,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
  GlyphSpinner,
  host,
  Input,
  TITLEBAR_AREAS,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const PERSONAL_ORGANIZATION_ID = '__personal__'
/*
const CREDITS_PURPLE = '#7c3aed'
*/

const LINK_URLS = {
  aiCredits: 'https://agent.unitalk.ai/spend',
  bookDemo: 'https://cal.com/severine/unitalk-demo',
  pricing: 'https://unitalk.ai/home/tarifs',
  settings: 'https://unitalk.ai/settings/common'
}

const CHECKOUT_URL = 'https://unitalk.ai/api/checkout/credits'
const CREDITS_DETAILS_URL = 'https://unitalk.ai/credits'
const CREDITS_PER_EURO = 500
const MAX_CUSTOM_AMOUNT = 9000
const MIN_CUSTOM_AMOUNT = 5
const PRESET_AMOUNTS = [5, 10, 25, 50, 100, 250]

const CREDITS_FORMAT = new Intl.NumberFormat('de-DE')
const EURO_FORMAT = new Intl.NumberFormat('de-DE', { currency: 'EUR', style: 'currency' })

const formatCredits = amount => CREDITS_FORMAT.format(amount * CREDITS_PER_EURO)
const formatEuro = amount => EURO_FORMAT.format(amount)

const clampAmount = value => {
  const parsed = Number(value)

  if (!Number.isFinite(parsed)) {
    return MIN_CUSTOM_AMOUNT
  }

  return Math.min(MAX_CUSTOM_AMOUNT, Math.max(MIN_CUSTOM_AMOUNT, parsed))
}

const ICON = 'shrink-0 text-(--ui-text-secondary)'

const messageOf = error => (error instanceof Error && error.message) || String(error || 'Request failed')

const initials = value => {
  const words = String(value || '?')
    .trim()
    .split(/\s+/)
    .filter(Boolean)

  return (words[0]?.[0] || '?').toUpperCase() + (words[1]?.[0] || '').toUpperCase()
}

const organizationImage = organization => organization?.imageUrl || null

function AccountAvatar({ className, imageUrl, label }) {
  return imageUrl
    ? jsx('img', {
        alt: '',
        className: cn('size-7 shrink-0 rounded-full object-cover', className),
        src: imageUrl
      })
    : jsx('span', {
        'aria-hidden': true,
        className: cn(
          'grid size-7 shrink-0 place-items-center rounded-full bg-(--ui-bg-quaternary) text-[0.6875rem] font-semibold text-(--ui-text-secondary)',
          className
        ),
        children: initials(label)
      })
}

function MenuActionRow({ icon, label, onSelect, trailing }) {
  return jsxs(DropdownMenuItem, {
    className: 'min-w-0 text-xs',
    onSelect,
    children: [icon, jsx('span', { className: 'min-w-0 flex-1 truncate', children: label }), trailing || null]
  })
}

function PublicMenuRows({ onBuyCredits, openLink }) {
  return jsxs('div', {
    className: 'flex flex-col gap-1',
    children: [
      jsx(MenuActionRow, {
        icon: jsx(Codicon, { className: ICON, name: 'settings', size: '0.875rem' }),
        key: 'settings',
        label: 'Settings',
        onSelect: () => openLink(LINK_URLS.settings)
      }),
      jsxs(MenuActionRow, {
        icon: jsx(Codicon, { className: ICON, name: 'zap', size: '0.875rem' }),
        key: 'ai-credits',
        label: 'AI Credits',
        onSelect: () => openLink(LINK_URLS.aiCredits)
      }),
      jsx(MenuActionRow, {
        icon: jsx(Codicon, { className: ICON, name: 'credit-card', size: '0.875rem' }),
        key: 'buy-credits',
        label: 'Buy credits',
        onSelect: onBuyCredits
      }),
      jsx(MenuActionRow, {
        icon: jsx(Codicon, { className: ICON, name: 'credit-card', size: '0.875rem' }),
        key: 'pricing',
        label: 'Pricing',
        onSelect: () => openLink(LINK_URLS.pricing)
      }),
      jsx(MenuActionRow, {
        icon: jsx(Codicon, { className: ICON, name: 'comment-discussion', size: '0.875rem' }),
        key: 'book-demo',
        label: 'Book a demo',
        onSelect: () => openLink(LINK_URLS.bookDemo)
      })
    ]
  })
}

/*
function AiCreditsValue() {
  return jsxs('span', {
    className: 'flex shrink-0 items-center gap-2',
    children: [
      jsx('span', {
        className: 'grid size-4 place-items-center rounded-full text-white',
        style: { backgroundColor: CREDITS_PURPLE },
        children: jsx(Codicon, { name: 'zap', size: '0.6rem' })
      }),
      jsx('span', {
        className: 'text-xs font-semibold tabular-nums',
        style: { color: CREDITS_PURPLE },
        children: '3379353'
      })
    ]
  })
}
*/

function CreateOrganizationDialog({ onOpenChange, open }) {
  const [name, setName] = useState('')
  const [slug, setSlug] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (open) {
      return
    }

    setName('')
    setSlug('')
    setError(null)
    setBusy(false)
  }, [open])

  const submit = async event => {
    event.preventDefault()

    if (busy || !name.trim()) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      await host.clerk.createOrganization({
        name: name.trim(),
        slug: slug.trim() || undefined
      })
      onOpenChange(false)
    } catch (err) {
      setError(messageOf(err))
    } finally {
      setBusy(false)
    }
  }

  return jsx(Dialog, {
    onOpenChange,
    open,
    children: jsx(DialogContent, {
      className: 'sm:max-w-md',
      children: jsxs('form', {
        className: 'flex flex-col gap-4',
        onSubmit: submit,
        children: [
          jsxs(DialogHeader, {
            children: [
              jsx(DialogTitle, { children: 'Create organization' }),
              jsx(DialogDescription, {
                children: 'Create a new organization and make it the active one.'
              })
            ]
          }),
          jsxs('div', {
            className: 'flex flex-col gap-2',
            children: [
              jsx(Input, {
                'aria-label': 'Organization name',
                autoFocus: true,
                disabled: busy,
                onChange: event => setName(event.target.value),
                placeholder: 'Organization name',
                value: name
              }),
              jsx(Input, {
                'aria-label': 'Organization slug',
                disabled: busy,
                onChange: event => setSlug(event.target.value),
                placeholder: 'Slug (optional)',
                value: slug
              }),
              error
                ? jsx('div', {
                    className: 'rounded-md bg-destructive/10 px-2 py-1.5 text-xs text-destructive',
                    children: error
                  })
                : null
            ]
          }),
          jsxs(DialogFooter, {
            children: [
              jsx(Button, {
                disabled: busy,
                onClick: () => onOpenChange(false),
                type: 'button',
                variant: 'outline',
                children: 'Cancel'
              }),
              jsx(Button, {
                disabled: busy || !name.trim(),
                type: 'submit',
                children: busy ? 'Creating…' : 'Create'
              })
            ]
          })
        ]
      })
    })
  })
}

function BuyCreditsDialog({ onOpenChange, onOpenUrl, open }) {
  const [amount, setAmount] = useState(PRESET_AMOUNTS[0])
  const [custom, setCustom] = useState(false)
  const [draft, setDraft] = useState(String(PRESET_AMOUNTS[0]))

  useEffect(() => {
    if (open) {
      return
    }

    setAmount(PRESET_AMOUNTS[0])
    setCustom(false)
    setDraft(String(PRESET_AMOUNTS[0]))
  }, [open])

  const effectiveAmount = custom ? clampAmount(draft) : amount

  const choosePreset = value => {
    setAmount(value)
    setCustom(false)
  }

  const startCustom = () => {
    setDraft(String(amount))
    setCustom(true)
  }

  const blurCustom = () => {
    const next = clampAmount(draft)

    setAmount(next)
    setDraft(String(next))
  }

  const buy = () => {
    const refCode =
      typeof window.promotekit_referral === 'string' ? `&refCode=${encodeURIComponent(window.promotekit_referral)}` : ''

    onOpenUrl(`${CHECKOUT_URL}?amount=${encodeURIComponent(effectiveAmount)}${refCode}`)
    onOpenChange(false)
  }

  return jsx(Dialog, {
    onOpenChange,
    open,
    children: jsxs(DialogContent, {
      children: [
        jsxs(DialogHeader, {
          className: 'items-center',
          children: [
            jsx('div', {
              className: 'mb-1 grid size-16 place-items-center rounded-full bg-primary text-primary-foreground',
              children: jsx(Codicon, { name: 'zap', size: '2rem' })
            }),
            jsx(DialogTitle, {
              className: 'w-full text-center',
              children: 'Buy AI credits'
            }),
            jsxs(DialogDescription, {
              className: 'text-center',
              children: [
                'Credits are the currency you use to chat, search the web, talk with your apps or generate images and video with any model in Unitalk ',
                jsx('button', {
                  className: 'cursor-pointer text-(--ui-blue) hover:underline',
                  onClick: () => onOpenUrl(CREDITS_DETAILS_URL),
                  type: 'button',
                  children: 'View details'
                })
              ]
            })
          ]
        }),
        jsx('div', {
          className: 'my-2 text-center text-3xl font-semibold text-primary',
          children: formatEuro(effectiveAmount)
        }),
        jsx('div', {
          className: 'grid grid-cols-3 gap-2',
          children: PRESET_AMOUNTS.map(value =>
            jsx(
              Button,
              {
                className: cn(
                  'h-auto w-full flex-col gap-1 py-3',
                  !custom &&
                    amount === value &&
                    'bg-(--ui-bg-quaternary) text-(--ui-text-primary) shadow-[inset_0_0_0_1.5px_var(--ui-purple)]'
                ),
                onClick: () => choosePreset(value),
                type: 'button',
                variant: 'outline',
                children: [
                  jsx('span', { className: 'text-base font-semibold', children: formatEuro(value) }),
                  jsx('span', {
                    className: 'text-[0.6875rem] text-primary',
                    children: `${formatCredits(value)} AI credits`
                  })
                ]
              },
              value
            )
          )
        }),
        custom
          ? jsx(Input, {
              'aria-label': 'Custom credit amount',
              autoFocus: true,
              className: 'w-full',
              max: MAX_CUSTOM_AMOUNT,
              min: MIN_CUSTOM_AMOUNT,
              onBlur: blurCustom,
              onChange: event => setDraft(event.target.value),
              placeholder: 'Enter an amount (5 – 9,000)',
              type: 'number',
              value: draft
            })
          : jsx(Button, {
              className: 'w-full',
              onClick: startCustom,
              type: 'button',
              variant: 'outline',
              children: 'Custom'
            }),
        jsx(DialogFooter, {
          children: jsx(Button, {
            className: 'w-full',
            onClick: buy,
            type: 'button',
            children: 'Buy credits'
          })
        })
      ]
    })
  })
}

function UserPanelContent({ onOpenUrl }) {
  const snapshot = useValue(host.clerk.snapshot)
  const [menuOpen, setMenuOpen] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [buyOpen, setBuyOpen] = useState(false)
  const [pendingOrganizationId, setPendingOrganizationId] = useState(null)
  const [organizationError, setOrganizationError] = useState(null)
  const [signingOut, setSigningOut] = useState(false)
  const [retryingSync, setRetryingSync] = useState(false)

  const openLink = url => {
    setMenuOpen(false)
    onOpenUrl(url)
  }

  if (snapshot.status === 'loading') {
    return jsx(Button, {
      'aria-label': 'Loading account',
      disabled: true,
      size: 'icon-titlebar',
      type: 'button',
      variant: 'ghost',
      children: jsx('span', { className: 'size-5 rounded-full bg-(--ui-bg-quaternary)' })
    })
  }

  const signedIn = snapshot.enabled && snapshot.status === 'signed-in' && Boolean(snapshot.user)

  // Signed out (or Clerk unavailable): keep the same panel, but replace the
  // organization switcher with a Login row that opens Clerk's sign-in modal.
  if (!signedIn) {
    const openSignIn = () => {
      if (typeof host.clerk.signIn !== 'function') {
        host.notifyError(new Error('Sign-in is not available in this app'), 'Could not open sign-in')
        return
      }

      void host.clerk.signIn().catch(error => host.notifyError(error, 'Could not open sign-in'))
    }

    return jsxs(DropdownMenu, {
      onOpenChange: setMenuOpen,
      open: menuOpen,
      children: [
        jsx(DropdownMenuTrigger, {
          asChild: true,
          children: jsx(Button, {
            'aria-label': 'Open account menu',
            className: 'text-(--ui-text-secondary)',
            size: 'icon-titlebar',
            style: { order: -1 },
            title: 'Sign in',
            type: 'button',
            variant: 'ghost',
            children: jsx(Codicon, { name: 'account', size: '1rem' })
          })
        }),
        jsxs(DropdownMenuContent, {
          align: 'start',
          className: 'min-w-52 max-w-72',
          side: 'bottom',
          sideOffset: 6,
          style: { width: '260px' },
          children: [
            jsx(MenuActionRow, {
              icon: jsx(Codicon, { className: ICON, name: 'sign-in', size: '0.875rem' }),
              key: 'sign-in',
              label: 'Log in',
              onSelect: () => {
                setMenuOpen(false)
                openSignIn()
              }
            }),
            jsx(DropdownMenuSeparator, {}),
            jsx(PublicMenuRows, {
              onBuyCredits: () => {
                setMenuOpen(false)
                setBuyOpen(true)
              },
              openLink
            })
          ]
        }),
        jsx(BuyCreditsDialog, {
          onOpenChange: setBuyOpen,
          onOpenUrl,
          open: buyOpen
        })
      ]
    })
  }

  const user = snapshot.user
  const activeOrganization = snapshot.activeOrganization
  const currentLabel = activeOrganization?.name || 'Personal account'
  const currentImage = activeOrganization ? organizationImage(activeOrganization) : user.imageUrl
  const pending = pendingOrganizationId !== null

  const switchTo = async organizationId => {
    setOrganizationError(null)
    setPendingOrganizationId(organizationId ?? PERSONAL_ORGANIZATION_ID)

    try {
      await host.clerk.setActiveOrganization(organizationId)
      setMenuOpen(false)
    } catch (error) {
      setOrganizationError(messageOf(error))
    } finally {
      setPendingOrganizationId(null)
    }
  }

  const signOut = async () => {
    setSigningOut(true)

    try {
      await host.clerk.signOut()
    } catch (error) {
      host.notifyError(error, 'Sign-out failed')
      setSigningOut(false)
    }
  }

  const retryTokenSync = async () => {
    setRetryingSync(true)

    try {
      await host.clerk.refresh()
    } catch (error) {
      host.notifyError(error, 'Could not refresh the account token')
    } finally {
      setRetryingSync(false)
    }
  }

  return jsxs(DropdownMenu, {
    onOpenChange: next => {
      setMenuOpen(next)

      if (!next) {
        setOrganizationError(null)
      }
    },
    open: menuOpen,
    children: [
      jsx(DropdownMenuTrigger, {
        asChild: true,
        children: jsx(Button, {
          'aria-label': `Open account menu for ${currentLabel}`,
          className: 'text-(--ui-text-secondary)',
          size: 'icon-titlebar',
          style: { order: -1 },
          title: currentLabel,
          type: 'button',
          variant: 'ghost',
          children: jsx(AccountAvatar, {
            className: 'size-5',
            imageUrl: currentImage,
            label: currentLabel
          })
        })
      }),
      jsxs(DropdownMenuContent, {
        align: 'start',
        className: 'min-w-52 max-w-72',
        side: 'bottom',
        sideOffset: 6,
        style: { width: '260px' },
        children: [
          jsxs(DropdownMenuSub, {
            children: [
              jsx(DropdownMenuSubTrigger, {
                hideChevron: true,
                children: [
                  jsx(AccountAvatar, {
                    className: 'size-5',
                    imageUrl: currentImage,
                    label: currentLabel
                  }),
                  jsx('span', { className: 'min-w-0 flex-1 truncate', children: currentLabel }),
                  jsx(Codicon, {
                    className: 'ml-auto shrink-0 text-(--ui-text-tertiary)',
                    name: 'chevron-down',
                    size: '0.75rem'
                  })
                ]
              }),
              jsxs(DropdownMenuSubContent, {
                className: 'min-w-48 max-w-72',
                sideOffset: 8,
                children: [
                  organizationError
                    ? jsx('div', {
                        className: 'mx-1 mb-1 rounded-md bg-destructive/10 px-2 py-1.5 text-xs text-destructive',
                        children: organizationError
                      })
                    : null,
                  jsxs('div', {
                    className: 'flex flex-col gap-1',
                    children: [
                      jsxs(DropdownMenuItem, {
                        className: 'min-w-0',
                        disabled: pending,
                        key: 'personal',
                        onSelect: event => {
                          event.preventDefault()
                          void switchTo(null)
                        },
                        children: [
                          jsx(AccountAvatar, {
                            className: 'size-5',
                            imageUrl: user.imageUrl,
                            label: user.fullName || user.email || user.username || 'Personal'
                          }),
                          jsx('span', {
                            className: 'min-w-0 flex-1 truncate',
                            children: 'Personal account'
                          }),
                          pendingOrganizationId === PERSONAL_ORGANIZATION_ID
                            ? jsx(GlyphSpinner, { className: 'ml-auto shrink-0' })
                            : !activeOrganization
                              ? jsx(Codicon, {
                                  className: 'ml-auto shrink-0 text-foreground',
                                  name: 'check',
                                  size: '0.75rem'
                                })
                              : null
                        ]
                      }),
                      ...snapshot.organizations.map(organization => {
                        const active = activeOrganization?.id === organization.id

                        return jsxs(
                          DropdownMenuItem,
                          {
                            className: 'min-w-0',
                            disabled: pending,
                            onSelect: event => {
                              event.preventDefault()
                              void switchTo(organization.id)
                            },
                            children: [
                              jsx(AccountAvatar, {
                                className: 'size-5',
                                imageUrl: organizationImage(organization),
                                label: organization.name
                              }),
                              jsx('span', {
                                className: 'min-w-0 flex-1 truncate',
                                children: organization.name
                              }),
                              pendingOrganizationId === organization.id
                                ? jsx(GlyphSpinner, { className: 'ml-auto shrink-0' })
                                : active
                                  ? jsx(Codicon, {
                                      className: 'ml-auto shrink-0 text-foreground',
                                      name: 'check',
                                      size: '0.75rem'
                                    })
                                  : null
                            ]
                          },
                          organization.id
                        )
                      })
                    ]
                  }),
                  jsx(DropdownMenuSeparator, {}),
                  jsxs(DropdownMenuItem, {
                    className: 'min-w-0',
                    disabled: pending,
                    onSelect: () => {
                      setMenuOpen(false)
                      setCreateOpen(true)
                    },
                    children: [
                      jsx(Codicon, { className: ICON, name: 'add', size: '0.875rem' }),
                      jsx('span', { className: 'min-w-0 flex-1 truncate', children: 'Create organization' })
                    ]
                  })
                ]
              })
            ]
          }),
          jsx(DropdownMenuSeparator, {}),
          jsx(PublicMenuRows, {
            onBuyCredits: () => {
              setMenuOpen(false)
              setBuyOpen(true)
            },
            openLink
          }),
          snapshot.tokenSyncError
            ? jsxs('div', {
                className: 'mx-2 my-1 rounded-md bg-(--ui-yellow)/10 px-2 py-1.5',
                children: [
                  jsx('div', {
                    className: 'text-xs text-(--ui-yellow)',
                    children: 'Organization switched, but the cloud credential did not sync.'
                  }),
                  jsx(Button, {
                    className: 'mt-1',
                    disabled: retryingSync,
                    onClick: () => void retryTokenSync(),
                    size: 'xs',
                    type: 'button',
                    variant: 'ghost',
                    children: retryingSync ? 'Retrying…' : 'Retry sync'
                  })
                ]
              })
            : null,
          snapshot.error
            ? jsx('div', {
                className: 'mx-2 my-1 rounded-md bg-destructive/10 px-2 py-1.5 text-xs text-destructive',
                children: snapshot.error
              })
            : null,
          jsx(DropdownMenuSeparator, {}),
          jsxs(DropdownMenuItem, {
            disabled: signingOut,
            onSelect: () => void signOut(),
            children: [
              jsx(Codicon, { className: ICON, name: 'sign-out', size: '0.875rem' }),
              jsx('span', { className: 'min-w-0 flex-1 truncate', children: signingOut ? 'Logging out…' : 'Log out' })
            ]
          })
        ]
      }),
      jsx(CreateOrganizationDialog, {
        onOpenChange: setCreateOpen,
        open: createOpen
      }),
      jsx(BuyCreditsDialog, {
        onOpenChange: setBuyOpen,
        onOpenUrl,
        open: buyOpen
      })
    ]
  })
}

function UserPanel({ onOpenUrl }) {
  return host.clerk ? jsx(UserPanelContent, { onOpenUrl }) : null
}

export default {
  id: 'user-panel',
  name: 'User Panel',
  description: 'Account and organization switcher for Unitalk Cloud.',
  register(ctx) {
    const openUrl = url => void ctx.os.openExternal(url)

    ctx.register({
      id: 'header',
      area: TITLEBAR_AREAS.center,
      render: () => jsx(UserPanel, { onOpenUrl: openUrl })
    })
  }
}
