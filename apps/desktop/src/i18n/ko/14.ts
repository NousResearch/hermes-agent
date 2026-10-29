// ko/14.ts — Korean translation of the `billing` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko14: TranslationOverrides = {
  settings: {
billing: {
      perMonth: amount => `${amount}/월`,
      creditsPerMonth: amount => `${amount} 크레딧/월`,
      usageLabel: label => `${label} 사용량`,
      freeTier: {
        signIn: '로그인',
        title: 'Nous 무료 등급을 사용 중이에요',
        message: 'Nous 계정으로 로그인하면 더 많은 모델과 도구를 사용할 수 있어요.',
        caption:
          'nous/welcome으로 실행되며 커넥터가 포함돼 있어요. 로그인하면 커넥터를 유지하면서 계정이 필요한 도구와 다른 모든 모델을 사용할 수 있어요.',
        name: 'Nous · 무료 등급',
        footnote:
          '무료 등급은 잔액도 결제도 없어요. Nous 계정으로 로그인하면 결제와 사용량이 표시돼요.',
        plan: '무료 등급',
        model: '모델',
        connectors: '커넥터',
        included: '포함됨'
      },
      amountValidation: {
        reloadTo: '충전 목표액',
        greaterThanThreshold: '충전 목표액은 임계값보다 커야 해요.',
        decimal: label => `${label}: 소수점 둘째 자리까지의 금액을 입력해 주세요.`,
        positive: label => `${label}: 금액은 $0보다 커야 해요.`,
        minimum: (label, amount) => `${label}: 최소 금액은 ${amount}예요.`,
        maximum: (label, amount) => `${label}: 최대 금액은 ${amount}예요.`
      },
      stepUp: {
        openVerification: '인증 페이지 열기',
        dismiss: '닫기',
        waiting: '인증 링크를 기다리는 중…',
        verify: '계속하려면 인증하세요',
        deniedTitle: '인증이 승인되지 않았어요',
        deniedBody: '이 터미널에 원격 결제를 허용하지 않은 채 인증이 끝났어요.',
        successTitle: '인증 완료',
        successBody: '이 터미널에서 원격 결제가 허용됐어요.'
      },
      charge: {
        added: amount => (amount ? `$${amount} 추가됐어요.` : '크레딧이 추가됐어요.'),
        failedTitle: '결제 실패',
        unconfirmedTitle: '결제 결과 미확인',
        unconfirmedBody: message =>
          `${message} 마지막 결제 결과가 확인되지 않았어요. 다시 시도하기 전에 잔액/내역을 확인해 주세요.`,
        checkTitle: '결제를 확인할 수 없어요',
        checkBody: '결제를 확인할 수 없어요.',
        untrackedTitle: '결제를 추적할 수 없어요',
        untrackedBody: '결제 서비스가 요청을 수락했지만 결제 ID를 반환하지 않았어요.',
        timeoutTitle: '5분이 지나도 처리 중이에요',
        timeoutBody: '결제가 아직 처리될 수 있어요. 다시 시도하기 전에 포털을 확인해 주세요.',
        authenticationRequired:
          '은행에서 인증(3DS)을 요구해요. 이 구매를 마치려면 포털에서 인증을 완료해 주세요.',
        expired: '카드가 만료됐어요. 포털에서 업데이트해 주세요.',
        declined: '카드가 거절됐어요. 포털에서 다른 카드로 시도해 보세요.',
        failedBody: reason => `결제가 처리되지 않았어요 (${reason}).`
      },
      title: '결제',
      preview: '미리보기',
      summary: {
        balance: '잔액',
        plan: '요금제',
        autoRefill: '자동 충전'
      },
      sections: {
        invoices: '청구서',

        plan: '요금제',
        paymentAndCredits: '결제 및 크레딧',
        usage: '사용량'
      },
      usage: {
        title: '사용량'
      },
      buyCredits: {
        customAmount: '크레딧 금액 직접 입력',
        title: '지금 크레딧 구매',
        buyButton: '구매',
        processing: '처리 중… 정산 확인 중',
        added: amount => `${amount} 추가됐어요. 잔액을 새로 고치는 중이에요.`,
        retry: '다시 시도',
        openPortal: '포털 열기'
      },
      plan: {
        title: '요금제',
        changePlan: '요금제 변경',
        viewPlans: '요금제 보기',
        backAria: '결제로 돌아가기',
        current: '현재 요금제',
        scheduled: '예약됨',
        empty: '지금 변경할 수 있는 요금제가 없어요.',
        undo: '실행 취소',
        undoing: '되돌리는 중…',
        downgrade: '다운그레이드',
        confirmDowngrade: '다운그레이드 확인',
        tryAgain: '다시 시도',
        checkingChange: '변경 사항 확인 중…',
        cannotChange: '여기서는 이 변경을 할 수 없어요.',
        alreadyOn: name => `${name} 요금제를 이미 사용 중이에요 — 변경할 것이 없어요.`,
        notScheduleable: '여기서는 이 변경을 예약할 수 없어요.',
        scheduling: '예약 중…',
        cancel: '취소',
        effectScheduled: (targetName, effectiveAt, creditsDelta) =>
          `${targetName} 요금제로 변경 — ${effectiveAt}에 적용돼요. 지금은 결제되지 않고, 그때까지 현재 요금제를 유지해요.${creditsDelta ? ` 월 크레딧 변경: ${creditsDelta}.` : ''}`
      },
      autoReload: {
        threshold: '임계값',
        thresholdAria: '자동 충전 임계값',
        reloadTo: '충전 목표액',
        reloadToAria: '자동 충전 목표 금액',
        turnOffConfirm: '자동 충전을 끌까요?',
        turnOff: '끄기',
        disable: '사용 안 함',
        updated: '자동 충전이 업데이트됐어요.',
        turnedOff: '자동 충전이 꺼졌어요.',
        manage: '관리',
        save: '저장',
        saving: '저장 중…',
        cancel: '취소'
      },
      state: {
        notice: {
          loggedOut: {
            title: 'Nous 계정 연결',
            message: 'Nous 계정으로 로그인하면 여기에서 잔액, 요금제, 사용량을 볼 수 있어요.',
            action: '로그인'
          },
          openPortal: '포털 열기 ↗',
          noCard: {
            title: '등록된 결제 수단이 없어요',
            message:
              '카드를 등록할 때까지 충전 크레딧 구매와 자동 충전은 사용할 수 없어요. 포털에서 카드를 추가해 주세요.',
            action: '카드 추가 ↗'
          }
        },
        paymentMethod: {
          title: '결제 수단',
          description: '충전과 구독 갱신에 사용할 카드를 관리하세요.',
          addAction: '결제 수단 추가',
          updateAction: '업데이트',
          provenance: {
            autoRefill: '자동 충전 카드',
            customerDefault: '고객 기본 카드',
            subPin: '구독 카드',
            suffix: label => ` - ${label}`
          }
        },
        buyCredits: {
          description: '카드로 한 번 결제하면 오늘 바로 잔액에 추가돼요.'
        },
        autoRefill: {
          title: '잔액이 부족하면 충전',
          genericDescription: '잔액이 임계값 아래로 떨어지면 자동으로 충전해 드려요.',
          offPill: '꺼짐',
          enabledPill: '켜짐',
          notAvailablePill: '—',
          manageCaption: '포털에서 자동 충전을 관리하세요.',
          turnOnCaption: '포털에서 자동 충전 켜기',
          chargesDescription: (reloadTo, threshold) =>
            `잔액이 ${threshold} 아래로 떨어지면 자동으로 ${reloadTo}를 결제해요.`,
          distinctCardCaption: cardLabel => `자동 충전은 ${cardLabel}로 결제돼요 — 포털에서 확인하세요`,
          distinctCardFallback: '다른 카드',
          reconcileAction: '확인 ↗'
        },
        usage: {
          subscriptionCredits: {
            title: '구독 크레딧',
            barLabel: '남은 구독 크레딧',
            captionResets: date => `${date}에 재설정`,
            valueOf: (remaining, monthly) => `${monthly} 중 ${remaining} 남음`,
            valueOver: (remaining, monthly, over) => `${monthly} 중 ${remaining} 남음 · ${over} 초과`
          },
          topupCredits: {
            title: '충전 크레딧',
            caption: '만료되지 않아요'
          },
          monthlyCap: {
            title: '월 지출 한도',
            barLabel: '사용한 월 지출 한도',
            captionDefault: '기본 한도',
            captionSpending: '월 원격 결제',
            valueUsed: (spent, limit) => `${limit} 중 ${spent} 사용`
          }
        },
        planCard: {
          freeTier: '무료',
          chooseAction: '선택 ↗',
          adjustPlanAction: '요금제 변경 ↗',
          unavailableCaption: '구독 정보를 사용할 수 없어요. 포털은 계속 열 수 있어요.',
          downgradeCaption: (tierName, when) => `${when}에 ${tierName} 요금제로 변경돼요.`,
          cancellationCaption: when => `${when}에 취소돼요.`,
          renewsCaption: date => `${date}에 갱신`,
          noSubscriptionCaption: '활성 구독이 없어요 — 유료 모델은 충전 크레딧에서 차감돼요.'
        }
      },
      errors: {
        consentRequired: {
          title: '카드 확인 필요',
          message: '포털에서 터미널 결제용으로 이 카드를 확인해 주세요'
        },
        insufficientScope: {
          title: '원격 결제 승인 필요',
          message: '원격 결제 허용이 필요해요. 충전을 시작해 허용한 다음 다시 시도해 주세요.'
        },
        remoteSpendingRevoked: {
          title: '원격 결제가 중지됐어요',
          messageByAdmin: '관리자가 이 터미널의 원격 결제를 중지했어요.',
          messageBySelf: '이 터미널의 원격 결제를 중지했어요.'
        },
        remoteSpendingReconnect: who => `${who} 이 기기를 다시 승인하려면 설정 -> 게이트웨이에서 다시 연결해 주세요.`,
        sessionRevoked: {
          title: '세션 로그아웃됨',
          message: '세션이 로그아웃됐어요. 설정 → 게이트웨이에서 다시 로그인해 주세요.'
        },
        cliBillingDisabled: {
          title: '원격 결제가 꺼져 있어요',
          message:
            '이 계정의 원격 결제가 꺼져 있어요 — 결제 관리자가 포털의 Hermes Agent 페이지에서 켤 수 있어요.'
        },
        roleRequired: {
          title: '관리자 권한 필요',
          message: '자금을 추가하려면 조직 관리자/소유자가 필요해요. 관리자에게 요청하거나 포털에서 관리하세요.'
        },
        idempotencyConflict: {
          title: '새 충전을 시작하세요',
          message: '🔴 해당 결제 키가 다른 금액에 이미 사용됐어요. 새 충전을 시작하세요.'
        },
        noPaymentMethod: {
          title: '저장된 카드 없음',
          message:
            '💳 아직 터미널 결제용으로 저장된 카드가 없어요. 포털에서 설정해 주세요 ' +
            '(일회성 크레딧 구매는 재사용 가능한 카드를 저장하지 않아요).'
        },
        orgAccessDenied: {
          title: '조직 접근이 거부됐어요',
          message: '이 토큰은 관리할 수 있는 조직에 연결되어 있지 않아요'
        },
        monthlyCapExceeded: {
          title: '월 지출 한도에 도달했어요',
          messageReached: '🔴 월 지출 한도에 도달했어요.',
          messageHeadroom: remaining => `🔴 월 지출 한도에 도달했어요 — $${remaining} 남았어요.`
        },
        rateLimited: {
          title: '지금 결제가 너무 많아요',
          message: mins =>
            mins > 0
              ? `🟡 지금 결제가 너무 많아요 (약 ${mins}분 후 다시 시도해 주세요). 결제 실패가 아니에요.`
              : '🟡 지금 결제가 너무 많아요. 결제 실패가 아니에요.'
        },
        stripeUnavailable: {
          title: 'Stripe에 문제가 발생했어요',
          message: mins =>
            mins > 0
              ? `Stripe에 문제가 발생했어요 — 약 ${mins}분 후 다시 시도해 주세요`
              : 'Stripe에 문제가 발생했어요 — 잠시 후 다시 시도해 주세요'
        },
        upgradeCapExceeded: {
          title: '일일 요금제 변경 한도에 도달했어요',
          message: '일일 요금제 변경 한도에 도달했어요 — 내일 다시 시도해 주세요'
        },
        endpointUnavailable: {
          title: '결제 엔드포인트를 사용할 수 없어요',
          message: '결제 엔드포인트가 JSON이 아닌 응답을 반환했어요 (이 배포에서는 사용할 수 없을 수 있어요).'
        },
        timeout: {
          title: '결제 요청 시간이 초과됐어요',
          message: '결제 요청 시간이 초과됐어요.'
        },
        transport: {
          title: '결제 연결에 실패했어요',
          message: '게이트웨이에 도달하기 전에 결제 요청이 실패했어요.'
        },
        default: {
          title: '결제 요청 실패',
          message: '결제 요청이 실패했어요.'
        }
      }
    },
  },
}
