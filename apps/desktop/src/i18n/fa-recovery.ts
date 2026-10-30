import type { TranslationOverrides } from './define-locale'

/** Copy for recovery, scheduled runs and browser permissions added upstream. */
export const faRecovery = {
  assistant: {
    thread: {
      errorChooseModelLink: 'انتخاب مدل دیگر',
      errorContinue: 'ادامه',
      errorCopyRaw: 'رونوشت خطا',
      errorDetailsCopy: 'رونوشت جزئیات',
      errorHideDetails: 'پنهان کردن جزئیات خطا',
      errorOpenModelSettings: 'باز کردن تنظیمات مدل',
      errorOpenProviderSettings: provider => `باز کردن تنظیمات ${provider}`,
      errorOpenSetupGuide: 'باز کردن راهنمای راه‌اندازی',
      errorRateLimitResetBody: (provider, time) =>
        `به سقف نرخ درخواست ${provider} رسیده‌اید. این محدودیت در ${time} بازنشانی می‌شود.`,
      errorRetryAtResetShort: time => `تلاش دوباره در ${time}`,
      errorSetupGuideLink: 'راهنمای راه‌اندازی',
      errorStartNewChatLink: 'شروع گفت‌وگوی جدید',
      errorCodes: {
        agent_init_failed: {
          title: 'شروع این گفت‌وگو ممکن نشد',
          body: provider =>
            `Hermes نتوانست برای این گفت‌وگو به ${provider} متصل شود. دوباره تلاش کنید؛ اگر مشکل ادامه داشت، ارائه‌دهنده را در تنظیمات بررسی کنید.`
        },
        local_runtime_unavailable: {
          title: 'سرور مدل محلی در دسترس نیست',
          body: 'سرور مدل محلی هنوز آماده نیست و ممکن است در حال راه‌اندازی باشد. کمی بعد دوباره تلاش کنید یا مدل دیگری انتخاب کنید.',
          bodyNoRetry:
            'سرور مدل محلی خاموش است. آن را از برنامهٔ دسکتاپ Hermes یا با دستور `hermes model` در پایانه راه‌اندازی کنید، یا مدل دیگری انتخاب کنید.'
        }
      }
    }
  },
  cron: {
    control: {
      stop: 'توقف',
      stopping: 'در حال توقف…',
      stopped: 'اجرا لغو شد',
      finishedFirst: 'اجرا پیش از توقف به پایان رسید',
      nothingToStop: 'اجرایی در جریان نیست',
      failedStop: 'توقف اجرا ممکن نشد',
      lastCancelled: reason => (reason ? `آخرین اجرا لغو شد: ${reason}` : 'آخرین اجرا لغو شد'),
      catchUpLabel: 'اجراهای ازدست‌رفته',
      catchUpHint: 'با اجراهایی که هنگام خاموش بودن Hermes از دست رفته‌اند چه کار شود.',
      catchUpChoices: { default: 'استفاده از پیش‌فرض', run: 'یک بار اجرا پس از بازگشت', skip: 'نادیده گرفتن' }
    }
  },
  settings: {
    notifications: {
      browserPermission: {
        title: 'اعلان‌های مرورگر',
        allow: 'اجازه دادن',
        states: {
          default:
            'هنگامی که زبانه در پس‌زمینه است، اعلان‌های این مرورگر نمایش داده شوند. تا وقتی اجازه ندهید، فقط عنوان زبانه چشمک می‌زند.',
          denied: 'در تنظیمات سایت این مرورگر مسدود شده است. عنوان زبانه همچنان چشمک می‌زند.',
          granted:
            'مجاز است. هنگامی که زبانه در پس‌زمینه است، اعلان‌ها نمایش داده می‌شوند و عنوان زبانه هم چشمک می‌زند.',
          insecure:
            'به HTTPS نیاز دارد؛ برای نمونه با Tailscale serve. روی HTTP معمولی، عنوان زبانه همچنان چشمک می‌زند.',
          unsupported: 'این مرورگر نمی‌تواند اعلان نمایش دهد. عنوان زبانه همچنان چشمک می‌زند.'
        }
      }
    },
    plugins: {
      installModal: {
        installAnyway: 'نصب با این حال',
        legacyBackendConfirm:
          'نسخهٔ پشتیبان از این برنامه قدیمی‌تر است و نمی‌تواند تضمین کند همان نسخه‌ای که بررسی کرده‌اید نصب شود: اگر مخزن از آن زمان تغییر کرده باشد، نسخهٔ جدیدتر نصب می‌شود. برای داشتن این تضمین، پشتیبان را به‌روز کنید.',
        sourceMoved: sha =>
          sha
            ? `مخزن از زمان بررسی شما تغییر کرده است. پیش از نصب، نسخهٔ جدید (commit ${sha}) را بررسی کنید.`
            : 'مخزن از زمان بررسی شما تغییر کرده است. در حال خواندن نسخهٔ جدید…',
        partialInstall: sha =>
          `${sha ? `بخش عامل این افزونه با commit ${sha} نصب شده است` : 'بخش عامل این افزونه نصب شده است'}، اما بخش دسکتاپ به‌دلیل تغییر مخزن پس از بررسی شما نصب نشد. برای جایگزینی هر دو بخش با نسخهٔ نمایش‌داده‌شده، «نصب مجدد اجباری» را در تنظیمات پیشرفته فعال کنید؛ یا با انصراف، فقط بخش عامل را نگه دارید.`
      }
    }
  },
  zones: { tabActions: title => `عملیات ${title}` }
} satisfies TranslationOverrides
