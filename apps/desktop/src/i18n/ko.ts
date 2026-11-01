import { mergeTranslations } from '@hermes/shared/i18n'

import { defineLocale, type TranslationOverrides } from './define-locale'
import { introKo } from './intro-ko'
import { ko01 } from './ko/01'
import { ko02 } from './ko/02'
import { ko03 } from './ko/03'
import { ko04 } from './ko/04'
import { ko05 } from './ko/05'
import { ko06 } from './ko/06'
import { ko07 } from './ko/07'
import { ko08 } from './ko/08'
import { ko09 } from './ko/09'
import { ko10 } from './ko/10'
import { ko11 } from './ko/11'
import { ko12 } from './ko/12'
import { ko13 } from './ko/13'
import { ko14 } from './ko/14'
import { ko15 } from './ko/15'
import { ko16 } from './ko/16'
import { ko17 } from './ko/17'
import { ko18 } from './ko/18'
import { ko19 } from './ko/19'
import { ko20 } from './ko/20'
import { ko21 } from './ko/21'
import { ko22 } from './ko/22'
import { ko23 } from './ko/23'
import { ko24 } from './ko/24'
import { ko25 } from './ko/25'
import { ko26 } from './ko/26'
import { ko27 } from './ko/27'
import { ko28 } from './ko/28'
import { ko29 } from './ko/29'
import { ko30 } from './ko/30'

// Korean is an overlay on English: every chunk contributes a partial subtree and
// `defineLocale` merges the lot over `en`, so a string added to en.ts later falls
// back to English instead of breaking the build.
const parts: TranslationOverrides[] = [
  { intro: introKo },
  ko01,
  ko02,
  ko03,
  ko04,
  ko05,
  ko06,
  ko07,
  ko08,
  ko09,
  ko10,
  ko11,
  ko12,
  ko13,
  ko14,
  ko15,
  ko16,
  ko17,
  ko18,
  ko19,
  ko20,
  ko21,
  ko22,
  ko23,
  ko24,
  ko25,
  ko26,
  ko27,
  ko28,
  ko29,
  ko30
]

export const koOverrides = parts.reduce<TranslationOverrides>(
  (acc, part) => mergeTranslations<TranslationOverrides>(acc, part),
  {}
)

export const ko = defineLocale(koOverrides)
