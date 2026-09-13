import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'

import { CONTROL_TEXT } from './constants'
import type { useVaultAdd } from './vault-settings-add'
import { IDENTIFIER_TYPES, type IdentifierType, VAULT_KINDS, type VaultKind } from './vault-settings-form'

interface VaultAddDialogProps {
  add: ReturnType<typeof useVaultAdd>
}

export function VaultAddDialog({ add }: VaultAddDialogProps) {
  const { t } = useI18n()
  const v = t.settings.vault
  const { addOpen, form, formError, setForm, closeAdd, submitAdd, addMutation } = add

  return (
    <Dialog onOpenChange={open => !open && closeAdd()} open={addOpen}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{v.addTitle}</DialogTitle>
          <DialogDescription>{v.addDescription}</DialogDescription>
        </DialogHeader>

        <form
          className="grid gap-4"
          onSubmit={e => {
            e.preventDefault()
            submitAdd()
          }}
        >
          <div className="grid items-start gap-4 sm:grid-cols-2">
            <Field htmlFor="vault-kind" label={v.kindField}>
              <Select onValueChange={value => setForm(f => ({ ...f, kind: value as VaultKind }))} value={form.kind}>
                <SelectTrigger className={CONTROL_TEXT} id="vault-kind">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {VAULT_KINDS.map(kind => (
                    <SelectItem key={kind} value={kind}>
                      {v.kinds[kind]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field htmlFor="vault-label" label={v.labelField}>
              <Input
                autoFocus
                id="vault-label"
                onChange={e => setForm(f => ({ ...f, label: e.target.value }))}
                placeholder={v.labelPlaceholder}
                value={form.label}
              />
            </Field>
          </div>

          <Field htmlFor="vault-origin" label={v.originField}>
            <Input
              id="vault-origin"
              inputMode="url"
              onChange={e => setForm(f => ({ ...f, origin: e.target.value }))}
              placeholder={form.kind === 'login' ? v.originPlaceholder : v.originPlaceholderCheckout}
              value={form.origin}
            />
          </Field>

          {form.kind === 'login' && (
            <>
              <div className="grid items-start gap-4 sm:grid-cols-2">
                <Field htmlFor="vault-id-type" label={v.identifierTypeField}>
                  <Select
                    onValueChange={value => setForm(f => ({ ...f, identifierType: value as IdentifierType }))}
                    value={form.identifierType}
                  >
                    <SelectTrigger className={CONTROL_TEXT} id="vault-id-type">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {IDENTIFIER_TYPES.map(type => (
                        <SelectItem key={type} value={type}>
                          {v.identifierTypes[type]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Field>
                <Field htmlFor="vault-identifier" label={v.identifierField}>
                  <Input
                    autoComplete="off"
                    id="vault-identifier"
                    onChange={e => setForm(f => ({ ...f, identifier: e.target.value }))}
                    value={form.identifier}
                  />
                </Field>
              </div>
              <Field htmlFor="vault-password" label={v.passwordField}>
                <Input
                  autoComplete="new-password"
                  id="vault-password"
                  onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                  type="password"
                  value={form.password}
                />
              </Field>
              <Field htmlFor="vault-otp" label={v.otpField} optional optionalLabel={v.optional}>
                <Input
                  autoComplete="off"
                  id="vault-otp"
                  onChange={e => setForm(f => ({ ...f, otpSecret: e.target.value }))}
                  placeholder={v.otpPlaceholder}
                  type="password"
                  value={form.otpSecret}
                />
                <p className="text-xs text-muted-foreground">{v.otpHint}</p>
              </Field>
            </>
          )}

          {form.kind === 'payment' && (
            <>
              <Field htmlFor="vault-card-number" label={v.cardNumberField}>
                <Input
                  autoComplete="off"
                  id="vault-card-number"
                  inputMode="numeric"
                  onChange={e => setForm(f => ({ ...f, cardNumber: e.target.value }))}
                  type="password"
                  value={form.cardNumber}
                />
              </Field>
              <Field htmlFor="vault-card-name" label={v.cardNameField}>
                <Input
                  autoComplete="off"
                  id="vault-card-name"
                  onChange={e => setForm(f => ({ ...f, cardName: e.target.value }))}
                  value={form.cardName}
                />
              </Field>
              <div className="grid items-start gap-4 sm:grid-cols-4">
                <Field htmlFor="vault-exp-month" label={v.expMonthField}>
                  <Input
                    id="vault-exp-month"
                    inputMode="numeric"
                    maxLength={2}
                    onChange={e => setForm(f => ({ ...f, expMonth: e.target.value }))}
                    placeholder="MM"
                    value={form.expMonth}
                  />
                </Field>
                <Field htmlFor="vault-exp-year" label={v.expYearField}>
                  <Input
                    id="vault-exp-year"
                    inputMode="numeric"
                    maxLength={4}
                    onChange={e => setForm(f => ({ ...f, expYear: e.target.value }))}
                    placeholder="YYYY"
                    value={form.expYear}
                  />
                </Field>
                <Field htmlFor="vault-cvc" label={v.cvcField}>
                  <Input
                    autoComplete="off"
                    id="vault-cvc"
                    inputMode="numeric"
                    maxLength={4}
                    onChange={e => setForm(f => ({ ...f, cvc: e.target.value }))}
                    type="password"
                    value={form.cvc}
                  />
                </Field>
                <Field htmlFor="vault-postal" label={v.postalField}>
                  <Input
                    id="vault-postal"
                    onChange={e => setForm(f => ({ ...f, postal: e.target.value }))}
                    value={form.postal}
                  />
                </Field>
              </div>
            </>
          )}

          {form.kind === 'address' && (
            <>
              <Field htmlFor="vault-line1" label={v.addressLine1Field}>
                <Input
                  id="vault-line1"
                  onChange={e => setForm(f => ({ ...f, line1: e.target.value }))}
                  value={form.line1}
                />
              </Field>
              <Field htmlFor="vault-line2" label={v.addressLine2Field} optional optionalLabel={v.optional}>
                <Input
                  id="vault-line2"
                  onChange={e => setForm(f => ({ ...f, line2: e.target.value }))}
                  value={form.line2}
                />
              </Field>
              <div className="grid items-start gap-4 sm:grid-cols-2">
                <Field htmlFor="vault-city" label={v.cityField}>
                  <Input
                    id="vault-city"
                    onChange={e => setForm(f => ({ ...f, city: e.target.value }))}
                    value={form.city}
                  />
                </Field>
                <Field htmlFor="vault-state" label={v.stateField} optional optionalLabel={v.optional}>
                  <Input
                    id="vault-state"
                    onChange={e => setForm(f => ({ ...f, state: e.target.value }))}
                    value={form.state}
                  />
                </Field>
              </div>
              <div className="grid items-start gap-4 sm:grid-cols-2">
                <Field htmlFor="vault-address-postal" label={v.postalField}>
                  <Input
                    id="vault-address-postal"
                    onChange={e => setForm(f => ({ ...f, postal: e.target.value }))}
                    value={form.postal}
                  />
                </Field>
                <Field htmlFor="vault-country" label={v.countryField}>
                  <Input
                    id="vault-country"
                    onChange={e => setForm(f => ({ ...f, country: e.target.value }))}
                    value={form.country}
                  />
                </Field>
              </div>
            </>
          )}

          {formError && <p className="text-xs text-destructive">{formError}</p>}

          <DialogFooter>
            <Button onClick={closeAdd} size="sm" type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={addMutation.isPending} size="sm" type="submit">
              {addMutation.isPending ? v.adding : v.addConfirm}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
