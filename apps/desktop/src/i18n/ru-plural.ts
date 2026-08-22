interface RussianPluralForms {
  one: string
  few: string
  many: string
  other?: string
}

const russianPluralRules = new Intl.PluralRules('ru')

export const pluralRu = (count: number, forms: RussianPluralForms): string => {
  const category = russianPluralRules.select(count)

  if (category === 'one') {
    return forms.one
  }

  if (category === 'few') {
    return forms.few
  }

  if (category === 'many') {
    return forms.many
  }

  return forms.other ?? forms.many
}

export const countRu = (count: number, forms: RussianPluralForms): string => {
  return `${count} ${pluralRu(count, forms)}`
}
