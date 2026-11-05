/** Machine-translated ko function leaves (Gemini 3.5 Flash-Lite). Params/logic identical to en; only display strings localized. */
import type { TranslationOverrides } from './define-locale'

export const koFnOverrides = {
  agents: {
    activeCount: count => `${count}\uAC1C \uD65C\uC131`,
    ageDays: days => `${days}\uC77C \uC804`,
    ageHours: hours => `${hours}\uC2DC\uAC04 \uC804`,
    ageMinutes: minutes => `${minutes}\uBD84 \uC804`,
    ageSeconds: seconds => `${seconds}\uCD08 \uC804`,
    agentsCount: count => `\uC5D0\uC774\uC804\uD2B8 ${count}\uAC1C`,
    delegation: index => `\uC704\uC784 ${index}`,
    durationMinutes: (minutes, seconds) => `${minutes}\uBD84 ${seconds}\uCD08`,
    durationSeconds: seconds => `${seconds}\uCD08`,
    failedCount: count => `${count}\uAC1C \uC2E4\uD328`,
    filesCount: count => `\uD30C\uC77C ${count}\uAC1C`,
    moreAgents: count => `\uC678 \uC5D0\uC774\uC804\uD2B8 ${count}\uAC1C \uB354`,
    moreFiles: count => `\uC678 \uD30C\uC77C ${count}\uAC1C \uB354`,
    tokens: value => `\uD1A0\uD070 ${value}\uAC1C`,
    toolsCount: count => `\uB3C4\uAD6C ${count}\uAC1C`,
    updatedAgo: age => `${age} \uC5C5\uB370\uC774\uD2B8\uB428`,
    workers: count => `\uC791\uC5C5\uC790 ${count}\uBA85`,
    workersActive: count => `\uD65C\uC131 ${count}\uBA85`
  },
  artifactCard: {
    generating: lines => `\uC0DD\uC131 \uC911\u2026 (${lines}\uC904)`,
    versionBadge: count => `\uBC84\uC804 ${count}\uAC1C`
  },
  artifactPreview: {
    versionOf: (current, total) => `v${total} \uC911 v${current}`
  },
  artifacts: {
    goToPage: (itemLabel, page) => `${itemLabel} \uD398\uC774\uC9C0 ${page}(\uC73C)\uB85C \uC774\uB3D9`,
    rangeOf: (start, end, total) => `\uC804\uCCB4 ${total}\uAC1C \uC911 ${start}-${end}`
  },
  assistant: {
    approval: {
      alwaysDescription: pattern =>
        `\uC601\uAD6C \uD5C8\uC6A9 \uBAA9\uB85D(~/.hermes/config.yaml)\uC5D0 \u201C${pattern}\u201D \uD328\uD134\uC774 \uCD94\uAC00\uB429\uB2C8\uB2E4. \uD604\uC7AC \uC138\uC158 \uBC0F \uD5A5\uD6C4 \uBAA8\uB4E0 \uC138\uC158\uC5D0\uC11C \uC774\uC640 \uAC19\uC740 \uBA85\uB839\uC5D0 \uB300\uD574 Hermes\uAC00 \uB2E4\uC2DC \uBB3B\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4.`
    },
    catalogInstall: {
      skill: name => `\uC2A4\uD0AC ${name}`,
      targetProfile: profile => `${profile} \uD504\uB85C\uD544\uC5D0 \uC124\uCE58\uD569\uB2C8\uB2E4`,
      envVar: name => `${name} 환경 변수`,
      serverNotConnected: (server, reason) => `MCP 서버 ${server}에 연결되지 않았습니다${reason ? `: ${reason}` : ''}`,
      missingEnv: names => `설정을 완료하려면 ${names}을(를) 설정하세요`
    },
    clarify: {
      questionProgress: (answered, total) => `\uCD1D ${total}\uAC1C \uC911 ${answered}\uAC1C \uB2F5\uBCC0 \uC644\uB8CC`
    },
    mcpSetup: {
      authorized: server => `${server} \uC778\uC99D\uB428`,
      enabled: server => `${server} \uD65C\uC131\uD654\uB428`,
      failed: server => `${server} \uC124\uC815 \uC2E4\uD328`,
      installed: server => `${server} \uC124\uCE58\uB428`,
      toolCount: count => `\uB3C4\uAD6C ${count}\uAC1C`
    },
    thread: {
      errorAuthKinds: {
        api_key: {
          body: provider =>
            `${provider}\uC5D0 \uC800\uC7A5\uB41C \uD0A4\uAC00 \uC798\uBABB\uB418\uC5C8\uAC70\uB098 \uCDE8\uC18C\uB418\uC5C8\uC2B5\uB2C8\uB2E4. \uC5C5\uB370\uC774\uD2B8 \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694.`,
          title: provider => `${provider}\uC5D0\uC11C API \uD0A4\uB97C \uAC70\uBD80\uD588\uC2B5\uB2C8\uB2E4`
        },
        oauth: {
          title: provider => `${provider} \uB85C\uADF8\uC778\uC774 \uB9CC\uB8CC\uB418\uC5C8\uC2B5\uB2C8\uB2E4`
        }
      },
      errorCodes: {
        auth: {
          body: provider =>
            `${provider}\uC5D0 \uC800\uC7A5\uB41C \uC790\uACA9 \uC99D\uBA85\uC774 \uC218\uB77D\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. \uC124\uC815\uC5D0\uC11C \uC218\uC815\uD558\uAC70\uB098 \uACF5\uAE09\uC790\uB97C \uC804\uD658\uD55C \uD6C4 \uBA54\uC2DC\uC9C0\uB97C \uB2E4\uC2DC \uBCF4\uB0B4\uC138\uC694.`,
          title: provider => `${provider}\uC5D0\uC11C \uB85C\uADF8\uC778\uC744 \uAC70\uBD80\uD588\uC2B5\uB2C8\uB2E4`
        },
        auth_permanent: {
          body: provider =>
            `${provider}\uC5D0 \uC800\uC7A5\uB41C \uC790\uACA9 \uC99D\uBA85\uC774 \uC798\uBABB\uB418\uC5C8\uAC70\uB098 \uCDE8\uC18C\uB418\uC5C8\uC2B5\uB2C8\uB2E4. \uC5C5\uB370\uC774\uD2B8\uD558\uAC70\uB098 \uACF5\uAE09\uC790\uB97C \uC804\uD658\uD55C \uD6C4 \uBA54\uC2DC\uC9C0\uB97C \uB2E4\uC2DC \uBCF4\uB0B4\uC138\uC694.`,
          title: provider => `${provider}\uC5D0\uC11C \uB85C\uADF8\uC778\uC744 \uAC70\uBD80\uD588\uC2B5\uB2C8\uB2E4`
        },
        billing: {
          body: provider =>
            `${provider} \uACC4\uC815\uC5D0 \uB0A8\uC740 \uD06C\uB808\uB527\uC774 \uC5C6\uC2B5\uB2C8\uB2E4. \uD06C\uB808\uB527\uC744 \uCDA9\uC804\uD558\uAC70\uB098 \uACF5\uAE09\uC790\uB97C \uC804\uD658\uD55C \uD6C4 \uB2E4\uC2DC \uBCF4\uB0B4\uC138\uC694.`
        },
        content_policy_blocked: {
          body: provider =>
            `${provider}\uC5D0\uC11C \uC774 \uBA54\uC2DC\uC9C0\uC5D0 \uC751\uB2F5\uD558\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4. \uB0B4\uC6A9\uC744 \uC218\uC815\uD558\uACE0 \uB2E4\uC2DC \uBCF4\uB0B4\uC138\uC694.`
        },
        empty_response: {
          body: provider =>
            `${provider}\uC5D0\uC11C \uC774 \uBA54\uC2DC\uC9C0\uC5D0 \uB300\uD55C \uC751\uB2F5\uC744 \uBC18\uD658\uD558\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. \uC7A0\uC2DC \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694.`
        },
        format_error: {
          body: provider =>
            `${provider}\uC5D0\uC11C \uC774 \uC694\uCCAD\uC758 \uD615\uC2DD\uC744 \uC218\uB77D\uD558\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. \uACF5\uAE09\uC790\uB97C \uC804\uD658\uD558\uAC70\uB098 \uC9C4\uB2E8 \uC815\uBCF4\uB97C \uBCF4\uB0B4\uC8FC\uC2DC\uBA74 \uD655\uC778\uD574 \uB4DC\uB9AC\uACA0\uC2B5\uB2C8\uB2E4.`
        },
        invalid_response: {
          body: provider =>
            `Hermes\uAC00 \uC77D\uC744 \uC218 \uC5C6\uB294 \uC751\uB2F5\uC744 ${provider}\uC5D0\uC11C \uBC18\uD658\uD588\uC2B5\uB2C8\uB2E4. \uC7A0\uC2DC \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694.`
        },
        model_not_found: {
          body: provider =>
            `${provider}\uC5D0\uC11C \uACC4\uC815\uC5D0 \uC774 \uBAA8\uB378\uC744 \uC81C\uACF5\uD558\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4. \uB2E4\uB978 \uBAA8\uB378\uC744 \uC120\uD0DD\uD55C \uD6C4 \uBA54\uC2DC\uC9C0\uB97C \uB2E4\uC2DC \uBCF4\uB0B4\uC138\uC694.`
        },
        overloaded: {
          body: provider =>
            `${provider}\uC5D0 \uBB38\uC81C\uAC00 \uBC1C\uC0DD\uD588\uC2B5\uB2C8\uB2E4. \uC7A0\uC2DC \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uAC70\uB098 \uACF5\uAE09\uC790\uB97C \uC804\uD658\uD558\uC138\uC694.`
        },
        provider_policy_blocked: {
          body: provider =>
            `${provider}\uC5D0\uC11C \uACC4\uC815\uC758 \uB370\uC774\uD130 \uB610\uB294 \uAC1C\uC778\uC815\uBCF4 \uBCF4\uD638 \uC124\uC815\uC5D0 \uB530\uB77C \uC774 \uC694\uCCAD\uC744 \uB77C\uC6B0\uD305\uD558\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4. \uB2E4\uB978 \uBAA8\uB378\uC744 \uC120\uD0DD\uD558\uAC70\uB098 \uACF5\uAE09\uC790\uB97C \uC804\uD658\uD558\uC138\uC694.`
        },
        rate_limit: {
          body: provider =>
            `${provider}\uC5D0\uC11C \uD604\uC7AC \uC694\uCCAD\uC744 \uC81C\uD55C\uD558\uACE0 \uC788\uC2B5\uB2C8\uB2E4. \uC7A0\uC2DC \uAE30\uB2E4\uB9B0 \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694.`
        },
        server_error: {
          body: provider =>
            `${provider}\uC5D0\uC11C \uC11C\uBC84 \uC624\uB958\uB97C \uBC18\uD658\uD588\uC2B5\uB2C8\uB2E4. \uC7A0\uC2DC \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uAC70\uB098 \uACF5\uAE09\uC790\uB97C \uC804\uD658\uD558\uC138\uC694.`
        },
        ssl_cert_verification: {
          body: provider =>
            `Hermes\uAC00 ${provider}\uC640\uC758 \uBCF4\uC548 \uC5F0\uACB0\uC744 \uD655\uC778\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4. \uB124\uD2B8\uC6CC\uD06C \uB610\uB294 \uD504\uB85D\uC2DC \uC124\uC815\uC744 \uD655\uC778\uD558\uAC70\uB098 \uACF5\uAE09\uC790\uB97C \uC804\uD658\uD55C \uD6C4 \uBA54\uC2DC\uC9C0\uB97C \uB2E4\uC2DC \uBCF4\uB0B4\uC138\uC694.`
        },
        timeout: {
          body: provider =>
            `${provider}\uC5D0 \uC5F0\uACB0\uD560 \uC218 \uC5C6\uAC70\uB098 \uC81C\uC2DC\uAC04\uC5D0 \uC751\uB2F5\uD558\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. \uC778\uD130\uB137 \uC5F0\uACB0\uC744 \uD655\uC778\uD55C \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694.`
        },
        upstream_blocked: {
          body: provider =>
            `${provider} \uC55E\uB2E8\uC758 \uBC29\uD654\uBCBD\uC774\uB098 CDN\uC774 \uBAA8\uB378\uC5D0 \uB3C4\uB2EC\uD558\uAE30 \uC804\uC5D0 \uC694\uCCAD\uC744 \uCC28\uB2E8\uD588\uC2B5\uB2C8\uB2E4. \uD0A4\uB294 \uC815\uC0C1\uC77C \uAC83\uC785\uB2C8\uB2E4. \uC124\uC815\uC758 \uACF5\uAE09\uC790 extra_headers\uB97C \uD1B5\uD574 User-Agent \uD5E4\uB354\uB97C \uC124\uC815\uD558\uAC70\uB098 \uACF5\uAE09\uC790\uB97C \uC804\uD658\uD55C \uD6C4 \uBA54\uC2DC\uC9C0\uB97C \uB2E4\uC2DC \uBCF4\uB0B4\uC138\uC694.`
        },
        upstream_rate_limit: {
          body: provider =>
            `${provider}\uC5D0\uC11C \uD604\uC7AC \uC694\uCCAD\uC744 \uC81C\uD55C\uD558\uACE0 \uC788\uC2B5\uB2C8\uB2E4. \uC7A0\uC2DC \uAE30\uB2E4\uB9B0 \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694.`
        }
      },
      errorLimitResets: time => `${time}\uC5D0 \uD55C\uB3C4\uAC00 \uCD08\uAE30\uD654\uB429\uB2C8\uB2E4`,
      errorOauthExpired: provider =>
        `${provider} \uB85C\uADF8\uC778\uC774 \uB9CC\uB8CC\uB418\uC5C8\uAC70\uB098 \uCDE8\uC18C\uB418\uC5C8\uC2B5\uB2C8\uB2E4. \uACC4\uC18D \uB300\uD654\uD558\uB824\uBA74 \uB2E4\uC2DC \uB85C\uADF8\uC778\uD558\uC138\uC694.`,
      errorRetryAtReset: time => `\uD55C\uB3C4\uAC00 \uCD08\uAE30\uD654\uB418\uBA74 \uC7AC\uC2DC\uB3C4 (${time})`,
      errorRetryScheduled: (time, wait) => `${time} (${wait} \uD6C4) \uC7AC\uC2DC\uB3C4 \uC911`,
      errorSignInAgain: provider => `${provider}\uC5D0 \uB2E4\uC2DC \uB85C\uADF8\uC778`,
      filesChanged: count => `\uBCC0\uACBD\uB418\uAC70\uB098 \uC218\uC815\uB41C \uD30C\uC77C ${count}\uAC1C`,
      loadingLocalModel: model => `${model}\uC744(\uB97C) \uBA54\uBAA8\uB9AC\uC5D0 \uB85C\uB4DC\uD558\uB294 \uC911`,
      resumeWhenBackgroundDone: count =>
        `\uBC31\uADF8\uB77C\uC6B4\uB4DC \uC791\uC5C5 ${count}\uAC1C\uAC00 \uC644\uB8CC\uB418\uBA74 \uC7AC\uAC1C\uB429\uB2C8\uB2E4`,
      thoughtFor: duration => `${duration} \uB3D9\uC548 \uC0DD\uAC01\uD568`,
      today: time => `\uC624\uB298, ${time}`,
      turnDuration: duration => `\uC774\uBC88 \uD134 \uC18C\uC694 \uC2DC\uAC04: ${duration}`,
      yesterday: time => `\uC5B4\uC81C, ${time}`
    },
    tool: {
      failedCalls: count => `\uB3C4\uAD6C \uD638\uCD9C ${count}\uAC1C \uC2E4\uD328`,
      failedMany: count => `\uB2E8\uACC4 ${count}\uAC1C \uC2E4\uD328`,
      recoveredMany: count => `\uC2E4\uD328\uD55C \uB2E8\uACC4 ${count}\uAC1C \uBCF5\uAD6C\uB428`,
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} \u201C${value}\u201D`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `${prefix.toLowerCase()} ${action.toLowerCase()} \uC2E4\uD589 \uC911`,
        runningTool: action => `${action.toLowerCase()} \uC2E4\uD589 \uC911`
      }
    }
  },
  billingBlock: {
    titleProvider: provider => `\uD06C\uB808\uB527 \uC18C\uC9C4 \u2014 ${provider}`
  },
  boot: {
    desktopBootFailedWithMessage: message => `\uB370\uC2A4\uD06C\uD1B1 \uBD80\uD305 \uC2E4\uD328: ${message}`,
    failure: {
      remoteSignInHint: signInLabel =>
        `\uC800\uC7A5\uB41C \uC6D0\uACA9 \uBE0C\uB77C\uC6B0\uC800 \uC138\uC158\uC5D0\uC11C \uB85C\uADF8\uC544\uC6C3\uD55C \uD6C4 ${signInLabel}\uC744(\uB97C) \uC5FD\uB2C8\uB2E4. \uBC88\uB4E4 \uBC31\uC5D4\uB4DC\uB85C \uC804\uD658\uD558\uB824\uBA74 \uB85C\uCEEC \uAC8C\uC774\uD2B8\uC6E8\uC774\uB97C \uC0AC\uC6A9\uD558\uC138\uC694.`,
      signInWithProvider: provider => `${provider}(\uC73C)\uB85C \uB85C\uADF8\uC778`
    },
    updateHold: {
      heldByProcess: pid =>
        `업데이트(프로세스 ${pid})는 종료되었지만, 그가 시작한 프로세스가 여전히 Hermes 설치를 점유하고 있습니다.`,
      since: time => `${time}부터 대기 중`,
      lastChecked: time => `마지막 확인: ${time}`
    }
  },
  commandCenter: {
    actions: count => `${count}\uAC1C\uC758 \uC791\uC5C5`,
    actualCost: cost => `\uC2E4\uC81C ${cost}`,
    days: count => `${count}\uC77C`,
    generatePet: {
      hatchRow: (_state, done, total) =>
        `${total}\uAC1C \uC911 ${done}\uBC88\uC9F8 \uD504\uB808\uC784 \uC2A4\uCF00\uCE58 \uC911\u2026`
    },
    hermesActiveSessions: (version, count) => `Hermes ${version} \xB7 \uD65C\uC131 \uC138\uC158 ${count}\uAC1C`,
    installTheme: {
      installs: count => `\uC124\uCE58 ${count}\uD68C`
    },
    maintenance: {
      actionFailed: name => `${name} \uC2DC\uC791 \uC2E4\uD328`,
      actionStarted: name => `${name} \uC2DC\uC791\uB428 \u2014 \uB85C\uADF8 \uCD94\uC801 \uC911...`,
      bytes: size => size,
      curatorLastRun: when => `\uB9C8\uC9C0\uB9C9 \uC2E4\uD589: ${when}`,
      memoryProvider: name => `\uD65C\uC131 \uC81C\uACF5\uC790: ${name}`,
      resetConfirm: target =>
        `${target}\uC744(\uB97C) \uC0AD\uC81C\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C? \uC774 \uC791\uC5C5\uC740 \uCDE8\uC18C\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      resetDone: files => `${files} \uC0AD\uC81C\uB428.`
    },
    newSessionInProject: project => `${project}\uC5D0\uC11C \uC0C8 \uC138\uC158 \uC2DC\uC791`,
    noUsage: period =>
      `\uCD5C\uADFC ${period}\uC77C \uB3D9\uC548 \uC0AC\uC6A9 \uB0B4\uC5ED\uC774 \uC5C6\uC2B5\uB2C8\uB2E4.`,
    openFolderAt: path => `\uD3F4\uB354\uB97C \uD504\uB85C\uC81D\uD2B8\uB85C \uC5F4\uAE30 \u2014 ${path}`,
    pets: {
      toggleFailed: enabled => `\uD3AB\uC744 ${enabled ? '\uCF1C' : '\uAEBC'}\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4.`
    },
    sharedGatewayRestartDescription: bots =>
      `\uC774 \uAE30\uAE30\uC758 \uBAA8\uB4E0 \uBD07\uC774 \uC7AC\uC5F0\uACB0\uB429\uB2C8\uB2E4: ${bots}`,
    sharedGatewayRestarted: count =>
      `\uACF5\uC720 \uAC8C\uC774\uD2B8\uC6E8\uC774\uAC00 \uC7AC\uC2DC\uC791\uB418\uC5C8\uC2B5\uB2C8\uB2E4 (\uBD07 ${count}\uAC1C)`,
    startInBranch: branch => `${branch}\uC5D0\uC11C \uC0C8 \uB300\uD654 \uC2DC\uC791`
  },
  connectors: {
    connectErrorFor: app => `${app} \uC778\uC99D\uC744 \uC2DC\uC791\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
    setup: server => `${server} \uC124\uC815`
  },
  connectorsPage: {
    card: {
      fact: {
        tools: count => `\uB3C4\uAD6C ${count}\uAC1C`,
        toolsOff: count => `\uB3C4\uAD6C ${count}\uAC1C \uAEBC\uC9D0`,
        toolsOn: count => `\uB3C4\uAD6C ${count}\uAC1C \uCF1C\uC9D0`,
        toolsSomeOn: (total, on) => `\uCD1D ${total}\uAC1C \uC911 ${on}\uAC1C \uCF1C\uC9D0`
      },
      kindPlugin: plugin => `MCP \xB7 \uD50C\uB7EC\uADF8\uC778 ${plugin}`,
      open: name => `${name} \uC5F4\uAE30`,
      turnServerOff: name => `${name}\uB044\uAE30`,
      turnServerOn: name => `${name} \uCF1C\uAE30`
    },
    dialog: {
      appSwitch: name => `Hermes\uC5D0\uC11C ${name} \uC0AC\uC6A9 \uAC00\uB2A5`,
      bothOn: name =>
        `\uB458 \uB2E4 \uCF1C\uC838 \uC788\uC5B4 Hermes\uAC00 ${name} \uB3C4\uAD6C\uB97C \uC911\uBCF5\uC73C\uB85C \uC778\uC2DD\uD569\uB2C8\uB2E4.`,
      disconnectTitle: name => `${name} \uC5F0\uACB0\uC744 \uD574\uC81C\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
      orgNote: count => `\uC870\uC9C1\uC5D0\uC11C \uB3C4\uAD6C ${count}\uAC1C\uB97C \uAED0\uC2B5\uB2C8\uB2E4.`,
      providedByPlugin: plugin => `\uD50C\uB7EC\uADF8\uC778 ${plugin}\uC5D0\uC11C \uC81C\uACF5`,
      removeServerTitle: name => `${name}\uC744(\uB97C) \uC81C\uAC70\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
      rulesAppOff: name =>
        `\uB3C4\uAD6C\uB97C \uBCC0\uACBD\uD558\uB824\uBA74 ${name}\uC744(\uB97C) \uCF1C\uC138\uC694.`,
      wayNotConnected: name =>
        `\uC544\uC9C1 \uC5F0\uACB0\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. \uBE0C\uB77C\uC6B0\uC800\uC5D0\uC11C ${name}\uC5D0 \uB85C\uADF8\uC778\uD558\uC138\uC694.`,
      waysTitle: name => `${name}\uC774(\uAC00) \uC2E4\uD589\uB418\uB294 \uC704\uCE58`
    },
    page: {
      matchesElsewhere: count =>
        `\uB2E4\uB978 \uADF8\uB8F9\uC5D0 ${count}\uAC1C\uC758 \uC77C\uCE58 \uD56D\uBAA9\uC774 \uB354 \uC788\uC2B5\uB2C8\uB2E4.`,
      segmentNoMatch: segment =>
        `${segment}\uC5D0 \uC77C\uCE58\uD558\uB294 \uD56D\uBAA9\uC774 \uC5C6\uC5B4 \uBAA8\uB4E0 \uD56D\uBAA9\uC774 \uD45C\uC2DC\uB429\uB2C8\uB2E4.`
    },
    searchPlaceholder: count => `\uC571 ${count}\uAC1C \uAC80\uC0C9`,
    tools: {
      categorySelect: count => `\uCE74\uD14C\uACE0\uB9AC ${count}\uAC1C`,
      conflictBody: (theyOff, theyOn) => {
        const they = [
          theyOff > 0 ? '\uCF1C\uB454 \uB3C4\uAD6C \uC911 ' + theyOff + '\uAC1C\uB97C \uAED0\uC2B5\uB2C8\uB2E4' : '',
          theyOn > 0
            ? '\uB044\uB824\uB358 \uB3C4\uAD6C \uC911 ' + theyOn + '\uAC1C\uB97C \uCF1C\uB450\uC5C8\uC2B5\uB2C8\uB2E4'
            : ''
        ].filter(Boolean)

        return (
          (they.length > 0
            ? '\uC0C1\uB300\uBC29\uC774 ' + they.join(', \uADF8\uB9AC\uACE0 ') + '\uD588\uC2B5\uB2C8\uB2E4. '
            : '') +
          '\uB0B4 \uBCC0\uACBD \uC0AC\uD56D\uC740 \uD654\uBA74\uC5D0 \uC720\uC9C0\uB418\uBA70, \uC800\uC7A5\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4.'
        )
      },
      facetSwitch: facet => `${facet} \uB3C4\uAD6C \uCF1C\uAE30/\uB044\uAE30`,
      footerDirty: (off, backOn) =>
        `\uB3C4\uAD6C ${off}\uAC1C \uAEBC\uC9D0, ${backOn === 0 ? '\uC5C6\uC74C' : backOn}\uAC1C \uB2E4\uC2DC \uCF1C\uC9D0`,
      goneTitle: name =>
        `${name}\uC774(\uAC00) \uCE74\uD0C8\uB85C\uADF8\uC5D0\uC11C \uC81C\uC678\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      hideDeprecated: count => `\uC9C0\uC6D0 \uC911\uB2E8\uB41C \uD56D\uBAA9 ${count}\uAC1C \uC228\uAE30\uAE30`,
      hideDetails: tool => `${tool}\uC758 \uAE30\uB2A5 \uC124\uBA85 \uC228\uAE30\uAE30`,
      moreHints: count => `+${count}`,
      needsAuthTitle: name =>
        `\uB3C4\uAD6C\uB97C \uBCF4\uB824\uBA74 ${name}\uC5D0 \uB85C\uADF8\uC778\uD558\uC138\uC694.`,
      offTitle: name => `${name}\uC774(\uAC00) \uAEBC\uC838 \uC788\uC2B5\uB2C8\uB2E4.`,
      searchCountPlaceholder: count => `\uB3C4\uAD6C ${count}\uAC1C \uAC80\uC0C9`,
      showAllTools: count => `\uC804\uCCB4 \uB3C4\uAD6C ${count}\uAC1C \uBCF4\uAE30`,
      showDeprecated: count => `\uC9C0\uC6D0 \uC911\uB2E8\uB41C \uD56D\uBAA9 ${count}\uAC1C \uBCF4\uAE30`,
      showDetails: tool => `${tool}\uC758 \uAE30\uB2A5 \uC124\uBA85 \uBCF4\uAE30`,
      summaryCount: count => `\uB3C4\uAD6C ${count}\uAC1C`,
      summaryPreviewTitle: name =>
        `\uC5F0\uACB0 \uD6C4 Hermes\uAC00 ${name}\uC5D0\uC11C \uC218\uD589\uD560 \uC218 \uC788\uB294 \uC791\uC5C5`,
      summarySomeOn: (on, total) => `\uCD1D ${total}\uAC1C \uC911 ${on}\uAC1C \uCF1C\uC9D0`,
      summaryTitle: name => `Hermes\uAC00 ${name}\uC5D0\uC11C \uC218\uD589\uD560 \uC218 \uC788\uB294 \uC791\uC5C5`,
      toolList: name => `${name} \uB3C4\uAD6C`,
      turnToolOff: tool => `${tool} \uB044\uAE30`,
      turnToolOn: tool => `${tool} \uCF1C\uAE30`
    }
  },
  cron: {
    count: count => `\uC791\uC5C5 ${count}\uAC1C`,
    dayFallback: value => `${value}\uC77C`,
    everyDayAt: time => `\uB9E4\uC77C ${time}`,
    everyDayOfWeekAt: (day, time) => `\uB9E4\uC8FC ${day} ${time}`,
    everyHourAt: minute => `\uB9E4\uC2DC\uAC04 ${minute}\uBD84`,
    monthlyOnDayAt: (dayOfMonth, time) => `\uB9E4\uC6D4 ${dayOfMonth}\uC77C ${time}`,
    weekdaysAt: time => `\uD3C9\uC77C ${time}`
  },
  desktop: {
    branchTitle: n => `\uCD08\uC548: \uBE0C\uB79C\uCE58 #${n}`,
    handoff: {
      failed: error => `\uD578\uB4DC\uC624\uD504 \uC2E4\uD328: ${error}`,
      success: platform =>
        `${platform}(\uC73C)\uB85C \uD578\uB4DC\uC624\uD504\uB418\uC5C8\uC2B5\uB2C8\uB2E4. \uC5B8\uC81C\uB4E0 \uC5EC\uAE30\uC11C \uC774\uC5B4\uC11C \uC9C4\uD589\uD558\uC138\uC694.`,
      systemNote: platform =>
        `\u21BB ${platform}(\uC73C)\uB85C \uD578\uB4DC\uC624\uD504\uB428 \u2014 \uC5B8\uC81C\uB4E0 \uC5EC\uAE30\uC11C \uC774\uC5B4\uC11C \uC9C4\uD589\uD558\uC138\uC694.`
    },
    hydrationSyncing: profile => `${profile} \uB3D9\uAE30\uD654 \uC911\u2026`,
    modelSwitchConfirmTitle: model => `${model}(\uC73C)\uB85C \uC804\uD658\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
    newChatsProfile: name =>
      `\uC0C8 \uB300\uD654\uC5D0\uC11C ${name} \uD504\uB85C\uD544\uC744 \uC0AC\uC6A9\uD569\uB2C8\uB2E4.`,
    noProfileNamed: (target, available) =>
      `"${target}" \uC774\uB984\uC758 \uD504\uB85C\uD544\uC774 \uC5C6\uC2B5\uB2C8\uB2E4. \uC0AC\uC6A9 \uAC00\uB2A5: ${available}`,
    profileStatus: current =>
      `\uD504\uB85C\uD544: ${current}. \uB2E4\uB978 \uD504\uB85C\uD544\uC5D0\uC11C \uB300\uD654\uB97C \uC2DC\uC791\uD558\uB824\uBA74 /profile <\uC774\uB984> \uB610\uB294 "\uC0C8 \uC138\uC158" \uC120\uD0DD\uAE30\uB97C \uC0AC\uC6A9\uD558\uC138\uC694.`,
    skillCommandsAvailable: count => `\uC2A4\uD0AC \uBA85\uB839\uC5B4 ${count}\uAC1C \uC0AC\uC6A9 \uAC00\uB2A5`,
    warningLine: message => `\uACBD\uACE0: ${message}`,
    yoloSystem: active => `\uD604\uC7AC \uC138\uC158 YOLO ${active ? '\uCF1C\uAE30' : '\uB044\uAE30'}`
  },
  fileMenu: {
    deleteTitle: name => `${name}\uC744(\uB97C) \uC0AD\uC81C\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`
  },
  freeTier: {
    busyBody: wait =>
      `Nous \uC11C\uBE44\uC2A4\uAC00 \uBC14\uBE60\uC11C Hermes\uAC00 \uB85C\uADF8\uC778 \uC644\uB8CC \uCC98\uB9AC\uB97C \uD558\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4. ${wait} \uD6C4\uC5D0 \uB2E4\uC2DC \uC2DC\uB3C4\uD574 \uC8FC\uC138\uC694. \uADF8\uB3D9\uC548 \uC138\uC158\uC740 \uADF8\uB300\uB85C \uC720\uC9C0\uB429\uB2C8\uB2E4.`,
    setupFailed: {
      rateLimited: wait =>
        `\uD604\uC7AC \uB9CE\uC740 \uC0AC\uC6A9\uC790\uAC00 \uC2DC\uC791\uD558\uB294 \uC911\uC774\uBBC0\uB85C Hermes\uAC00 ${wait} \uD6C4\uC5D0 \uB2E4\uC2DC \uC2DC\uB3C4\uD569\uB2C8\uB2E4. \uB85C\uADF8\uC778\uC740 \uBB34\uB8CC\uC774\uBA70 \uB300\uAE30 \uC2DC\uAC04\uC774 \uC5C6\uC2B5\uB2C8\uB2E4.`
    },
    signedInAs: email => `${email}(\uC73C)\uB85C \uB85C\uADF8\uC778\uB428`,
    statusLabel: model => `Nous \xB7 ${model}`
  },
  guidedGreeting: {
    nameSuggestion: name =>
      `(\uC6D0\uD558\uC2E0\uB2E4\uBA74 \uADF8\uB0E5 ${name}(\uC774)\uB77C\uACE0 \uBD80\uB97C \uC218\uB3C4 \uC788\uC2B5\uB2C8\uB2E4.)`
  },
  install: {
    authNeedsOauth: provider =>
      `\uC774 \uAC8C\uC774\uD2B8\uC6E8\uC774\uB97C \uD14C\uC2A4\uD2B8\uD558\uAE30 \uC804\uC5D0 ${provider}(\uC73C)\uB85C \uB85C\uADF8\uC778\uD558\uC138\uC694.`,
    currentStage: stage => ` -- \uD604\uC7AC: ${stage}`,
    lines: count => `${count}\uC904`,
    progress: (completed, total) => `\uCD1D ${total}\uB2E8\uACC4 \uC911 ${completed}\uB2E8\uACC4 \uC644\uB8CC`,
    signInWith: provider => `${provider}(\uC73C)\uB85C \uB85C\uADF8\uC778`,
    testSucceeded: (baseUrl, version) =>
      `${baseUrl}${version ? ` (${version})` : ''}\uC5D0 \uC5F0\uACB0\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
    unsupportedDesc: platform =>
      `${platform}\uC5D0\uC11C\uB294 \uC544\uC9C1 \uC790\uB3D9 \uCD5C\uCD08 \uC2E4\uD589 \uC124\uCE58\uB97C \uC0AC\uC6A9\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4. \uD130\uBBF8\uB110\uC744 \uC5F4\uACE0 \uC544\uB798 \uBA85\uB839\uC5B4\uB97C \uC2E4\uD589\uD55C \uB2E4\uC74C \uC571\uC744 \uB2E4\uC2DC \uC2DC\uC791\uD558\uC138\uC694. \uC774\uD6C4 \uC2E4\uD589 \uC2DC\uC5D0\uB294 \uC774 \uB2E8\uACC4\uB97C \uAC74\uB108\uB701\uB2C8\uB2E4.`
  },
  keybinds: {
    conflictWith: label => `\u201C${label}\u201D\uC5D0 \uC774\uBBF8 \uC9C0\uC815\uB428`,
    subtitle: open =>
      `\uB2E8\uCD95\uD0A4\uB97C \uD074\uB9AD\uD558\uC5EC \uC7AC\uD560\uB2F9 \xB7 ${open}\uC744(\uB97C) \uB204\uB974\uBA74 \uC774 \uD328\uB110\uC774 \uB2E4\uC2DC \uC5F4\uB9BD\uB2C8\uB2E4.`
  },
  messaging: {
    advanced: count => `\uACE0\uAE09 (${count})`,
    approvedUser: name => `${name} \uC2B9\uC778\uB428`,
    approvedUsers: count => `\uC2B9\uC778\uB41C \uC0AC\uC6A9\uC790 (${count})`,
    clearField: key => `${key} \uC9C0\uC6B0\uAE30`,
    disableAria: name => `${name} \uBE44\uD65C\uC131\uD654`,
    enableAria: name => `${name} \uD65C\uC131\uD654`,
    failedApprove: name => `${name} \uC2B9\uC778 \uC2E4\uD328`,
    failedClear: key => `${key} \uC9C0\uC6B0\uAE30 \uC2E4\uD328`,
    failedRevoke: name => `${name} \uAD8C\uD55C \uD574\uC81C \uC2E4\uD328`,
    failedSave: name => `${name} \uC800\uC7A5 \uC2E4\uD328`,
    failedUpdate: name => `${name} \uC5C5\uB370\uC774\uD2B8 \uC2E4\uD328`,
    keyCleared: key => `${key} \uC9C0\uC6CC\uC9D0`,
    pendingAria: count => `\uB300\uAE30 \uC911\uC778 \uD398\uC5B4\uB9C1 \uC694\uCCAD ${count}\uAC1C`,
    pendingRequests: count => `\uB300\uAE30 \uC911\uC778 \uC694\uCCAD (${count})`,
    platformDisabled: name => `${name} \uBE44\uD65C\uC131\uD654\uB428`,
    platformEnabled: name => `${name} \uD65C\uC131\uD654\uB428`,
    revokeAria: name => `${name} \uAD8C\uD55C \uD574\uC81C`,
    revokeDesc: name =>
      `${name}\uB2D8\uC740 \uB2E4\uC74C \uBA54\uC2DC\uC9C0\uBD80\uD130 \uC561\uC138\uC2A4 \uAD8C\uD55C\uC744 \uC783\uACE0 \uB354 \uC774\uC0C1 \uC778\uC2DD\uB418\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4.`,
    revokedUser: name => `${name} \uAD8C\uD55C \uD574\uC81C\uB428`,
    setupSaved: name => `${name} \uC124\uC815\uC774 \uC800\uC7A5\uB428`,
    setupUpdated: name => `${name} \uC124\uC815\uC774 \uC5C5\uB370\uC774\uD2B8\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
    telegramQr: {
      expiresIn: remaining => `${remaining} \uD6C4 \uB9CC\uB8CC`,
      savedRestartFailed: detail =>
        `\uD154\uB808\uADF8\uB7A8 \uC124\uC815\uC774 \uC800\uC7A5\uB418\uC5C8\uC73C\uB098, \uAC8C\uC774\uD2B8\uC6E8\uC774 \uC7AC\uC2DC\uC791\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4${detail}`,
      stillWaiting: detail =>
        `\uD154\uB808\uADF8\uB7A8\uC744 \uACC4\uC18D \uAE30\uB2E4\uB9AC\uB294 \uC911\uC785\uB2C8\uB2E4. \uC7AC\uC2DC\uB3C4 \uAC04\uACA9: ${detail}`
    },
    waitingSince: minutes => (minutes < 1 ? '\uBC29\uAE08 \uC804' : `${minutes}\uBD84 \uC804`)
  },
  notifications: {
    mcp: {
      disableFailed: name => `${name} MCP\uB97C \uBE44\uD65C\uC131\uD654\uD558\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4.`,
      disabledMessage: name =>
        `${name} MCP\uAC00 \uBE44\uD65C\uC131\uD654\uB418\uC5C8\uC2B5\uB2C8\uB2E4. [\uAE30\uB2A5] \u2192 [MCP]\uC5D0\uC11C \uC5B8\uC81C\uB4E0\uC9C0 \uB2E4\uC2DC \uD65C\uC131\uD654\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4.`,
      errorMessage: name => `${name} MCP \uC0C1\uD0DC \uAC80\uC0AC\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
      needsAuthMessage: name => `${name} MCP \uC7AC\uC778\uC99D\uC774 \uD544\uC694\uD569\uB2C8\uB2E4.`
    },
    more: count => `\uC54C\uB9BC ${count}\uAC1C \uB354`,
    native: {
      approvalTitleNamed: session => `\uC2B9\uC778 \uD544\uC694 \u2014 ${session}`,
      inputTitleNamed: session => `\uC785\uB825 \uD544\uC694 \u2014 ${session}`
    },
    updateReadyMessage: count => `\uC0C8\uB85C\uC6B4 \uBCC0\uACBD\uC0AC\uD56D ${count}\uAC1C \uC0AC\uC6A9 \uAC00\uB2A5`,
    voice: {
      liveUnavailable: reason =>
        `GPT-Live \uC74C\uC131 \uCC44\uD305\uC744 \uC0AC\uC6A9\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4: ${reason}. \uB300\uC2E0 \uC74C\uC131\uC778\uC2DD\uC744 \uC0AC\uC6A9\uD569\uB2C8\uB2E4.`,
      sayStopToEnd: phrase =>
        `\uC74C\uC131 \uCC44\uD305\uC744 \uC885\uB8CC\uD558\uB824\uBA74 "${phrase}"\uB77C\uACE0 \uB9D0\uC500\uD558\uC138\uC694.`
    }
  },
  preview: {
    binaryBody: label =>
      `${label}\uC744(\uB97C) \uBBF8\uB9AC \uBCF4\uBA74 \uC77D\uC744 \uC218 \uC5C6\uB294 \uD14D\uC2A4\uD2B8\uAC00 \uD45C\uC2DC\uB420 \uC218 \uC788\uC2B5\uB2C8\uB2E4.`,
    console: {
      messages: count => `\uCF58\uC194 \uBA54\uC2DC\uC9C0 ${count}\uAC1C`,
      selected: count => `\uC120\uD0DD\uB428 ${count}\uAC1C`,
      sentMessage: count => `\uB85C\uADF8 \uD56D\uBAA9 ${count}\uAC1C\uAC00 \uC791\uC131\uAE30\uC5D0 \uCD94\uAC00\uB428`
    },
    largeBody: (label, size) =>
      `${label}\uC758 \uD06C\uAE30\uB294 ${size}\uC785\uB2C8\uB2E4. Hermes\uB294 \uCC98\uC74C 512KB\uB9CC \uD45C\uC2DC\uD569\uB2C8\uB2E4.`,
    missingBody: label =>
      `${label}이(가) 삭제되었거나 이동되었거나 임시 위치가 지워졌습니다. 다음 실행 시 이 탭은 복원되지 않습니다.`,
    noInlineBody: mimeType =>
      `${mimeType || '\uC774 \uD30C\uC77C \uD615\uC2DD'}\uC740(\uB294) \uCEE8\uD14D\uC2A4\uD2B8\uB85C \uACC4\uC18D \uCCA8\uBD80\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4.`,
    saveFailed: message => `\uC800\uC7A5\uD558\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4: ${message}`,
    web: {
      addComments: count =>
        count === 1 ? '\uB313\uAE00 1\uAC1C \uCD94\uAC00' : `\uB313\uAE00 ${count}\uAC1C \uCD94\uAC00`,
      commentTitle: n => `\uB313\uAE00 ${n}`,
      failedRestarting: message => `\uC11C\uBC84 \uC7AC\uC2DC\uC791 \uC2E4\uD328: ${message}`,
      fileChanged: url =>
        `\uD30C\uC77C\uC774 \uBCC0\uACBD\uB418\uC5B4 \uBBF8\uB9AC\uBCF4\uAE30\uB97C \uB2E4\uC2DC \uBD88\uB7EC\uC624\uB294 \uC911: ${url}`,
      filesChanged: (count, url) =>
        `\uD30C\uC77C ${count}\uAC1C \uBCC0\uACBD\uB428, \uBBF8\uB9AC\uBCF4\uAE30 \uB2E4\uC2DC \uBD88\uB7EC\uC624\uB294 \uC911: ${url}`,
      finishedRestarting: message =>
        `Hermes\uAC00 \uBBF8\uB9AC\uBCF4\uAE30 \uC11C\uBC84 \uC7AC\uC2DC\uC791\uC744 \uC644\uB8CC\uD588\uC2B5\uB2C8\uB2E4${message ? `: ${message}` : ''}`,
      loadFailedConsole: (code, message) => `\uB85C\uB4DC \uC2E4\uD328${code ? ` (${code})` : ''}: ${message}`,
      lookingRestart: taskId =>
        `Hermes\uAC00 \uC7AC\uC2DC\uC791\uD560 \uBBF8\uB9AC\uBCF4\uAE30 \uC11C\uBC84\uB97C \uCC3E\uB294 \uC911 (${taskId})`,
      openTarget: url => `${url} \uC5F4\uAE30`,
      startRestartFailed: message =>
        `\uC11C\uBC84 \uC7AC\uC2DC\uC791\uC744 \uC2DC\uC791\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4: ${message}`,
      watchFailed: message =>
        `\uBBF8\uB9AC\uBCF4\uAE30 \uD30C\uC77C\uC744 \uAC10\uC2DC\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4: ${message}`
    }
  },
  profiles: {
    count: count => `\uD504\uB85C\uD544 ${count}\uAC1C`,
    defaultSet: name =>
      `${name}\uC774(\uAC00) \uAE30\uBCF8\uAC12\uC73C\uB85C \uC124\uC815\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
    fleet: {
      deleteOn: gateway => ` (${gateway} \uAE30\uC900)`,
      gateway: gateway => `${gateway}\uC758 \uD504\uB85C\uD544`,
      gatewayUnreachable: gateway => `${gateway} \xB7 \uC5F0\uACB0\uD560 \uC218 \uC5C6\uC74C`,
      onGateway: (name, gateway) => `${name} \xB7 ${gateway}`,
      switchTo: (name, gateway) => `${gateway}\uC758 ${name}(\uC73C)\uB85C \uC804\uD658`
    },
    invalidName: hint => `\uC798\uBABB\uB41C \uC774\uB984\uC785\uB2C8\uB2E4. ${hint}`,
    remoteOverride: {
      authFailedMessage: (profile, host) =>
        `${host}\uC5D0\uC11C ${profile}\uC5D0 \uB300\uD574 \uC800\uC7A5\uB41C \uD1A0\uD070\uC744 \uAC70\uBD80\uD588\uC2B5\uB2C8\uB2E4. \uC6D0\uACA9 \uCE21\uC5D0\uC11C \uD1A0\uD070\uC774 \uBCC0\uACBD\uB418\uC5C8\uC744 \uC218 \uC788\uC2B5\uB2C8\uB2E4.`,
      badge: host => `${host}\uC5D0\uC11C \uC2E4\uD589 \uC911`,
      collisionWarning: label =>
        `\uC124\uC815\uC5D0 \uC774\uBBF8 \u201C${label}\u201D(\uC774)\uB77C\uB294 \uC774\uB984\uC758 \uAC8C\uC774\uD2B8\uC6E8\uC774\uAC00 \uC874\uC7AC\uD569\uB2C8\uB2E4. \uC774 \uD504\uB85C\uD544 \uC5F0\uACB0\uC740 \uBCC4\uB3C4\uB85C \uC720\uC9C0\uB418\uBA70 \uBCC0\uACBD\uB418\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4.`,
      confirmNote: (profile, host) =>
        `${profile}\uC758 \uC0C8 \uCC44\uD305\uC740 ${host}\uC5D0\uC11C \uC2E4\uD589\uB429\uB2C8\uB2E4. \uD574\uB2F9 \uCEF4\uD4E8\uD130\uC5D0\uC11C \uBA85\uB839\uC5B4\uAC00 \uC2E4\uD589\uB418\uACE0 \uD30C\uC77C\uC774 \uC77D\uD788\uBA70, \uC774 \uCEF4\uD4E8\uD130\uC5D0\uC11C\uB294 \uC2E4\uD589\uB418\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4. \uC2E0\uB8B0\uD560 \uC218 \uC788\uB294 \uD638\uC2A4\uD2B8\uC5D0\uB9CC \uC5F0\uACB0\uD558\uC138\uC694.`,
      removedMessage: profile =>
        `\uC774\uC81C ${profile}\uC774(\uAC00) \uC774 \uCEF4\uD4E8\uD130\uC5D0\uC11C \uC2E4\uD589\uB429\uB2C8\uB2E4`,
      savedMessage: (profile, host) =>
        `\uC774\uC81C ${profile}\uC774(\uAC00) ${host}\uC5D0\uC11C \uC2E4\uD589\uB429\uB2C8\uB2E4`,
      title: profile => `${profile}\uC744(\uB97C) \uC6D0\uACA9 \uD638\uC2A4\uD2B8\uC5D0 \uC5F0\uACB0`
    },
    setColor: color => `\uC0C9\uC0C1 ${color} \uC124\uC815`,
    skills: count => `\uC2A4\uD0AC ${count}\uAC1C`,
    soulPlaceholder:
      mode => `\uC774 \uD504\uB85C\uD544\uC758 \uC2DC\uC2A4\uD15C \uD504\uB86C\uD504\uD2B8 / \uD398\uB974\uC18C\uB098\uC785\uB2C8\uB2E4.
${mode} \uAE30\uBCF8\uAC12\uC744 \uC720\uC9C0\uD558\uB824\uBA74 \uBE44\uC6CC \uB450\uC138\uC694.`,
    status: {
      needsInput: count => `\uB2F5\uBCC0\uC774 \uD544\uC694\uD55C \uC138\uC158 ${count}\uAC1C`,
      unread: count => `\uC77D\uC9C0 \uC54A\uC740 \uC138\uC158 ${count}\uAC1C`,
      working: count => `\uC2E4\uD589 \uC911\uC778 \uC138\uC158 ${count}\uAC1C`
    },
    switchConnectionFailed: name => `${name}\uC5D0 \uC5F0\uACB0\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4`,
    switchToConnection: name => `${name}(\uC73C)\uB85C \uC804\uD658`,
    switchToProfile: name => `${name}(\uC73C)\uB85C \uC804\uD658`
  },
  prompts: {
    vaultCodeDesc: site =>
      `${site}\uC5D0\uC11C \uC77C\uD68C\uC6A9 \uCF54\uB4DC(\uBB38\uC790\uBA54\uC2DC\uC9C0, \uC774\uBA54\uC77C \uB610\uB294 \uC778\uC99D \uC571)\uB97C \uC694\uCCAD\uD558\uACE0 \uC788\uC2B5\uB2C8\uB2E4. \uC5EC\uAE30\uC5D0 \uC785\uB825\uD558\uBA74 Hermes\uAC00 \uD398\uC774\uC9C0\uC5D0 \uC785\uB825\uD558\uBA70, \uBAA8\uB378\uC740 \uC774 \uCF54\uB4DC\uB97C \uBCFC \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
    vaultCodeTitle: site => `${site} \uC778\uC99D \uCF54\uB4DC`,
    vaultSaveDesc: origin =>
      `Hermes\uAC00 ${origin}\uC758 \uB85C\uADF8\uC778 \uD398\uC774\uC9C0\uC5D0 \uB3C4\uB2EC\uD588\uC9C0\uB9CC \uC800\uC7A5\uB41C \uB85C\uADF8\uC778\uC774 \uC5C6\uC2B5\uB2C8\uB2E4. \uC5EC\uAE30\uC5D0 \uD55C \uBC88\uB9CC \uC785\uB825\uD558\uC138\uC694. \uC774 \uAE30\uAE30\uC5D0 \uC554\uD638\uD654\uB418\uC5B4 \uC800\uC7A5\uB418\uBA70 \uBAA8\uB378\uC774 \uBE44\uBC00\uBC88\uD638\uB97C \uC804\uD600 \uBCF4\uC9C0 \uC54A\uACE0 \uD398\uC774\uC9C0\uC5D0 \uC785\uB825\uB429\uB2C8\uB2E4.`,
    vaultSaveTitle: site => `${site} \uB85C\uADF8\uC778\uC744 \uC800\uC7A5\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
    vaultUnlockDesc: name =>
      `${name}\uC5D0 \uC800\uC7A5\uB41C \uB85C\uADF8\uC778\uC73C\uB85C \uC5D0\uC774\uC804\uD2B8\uAC00 \uC0AC\uC774\uD2B8\uC5D0 \uB85C\uADF8\uC778\uD558\uB824\uACE0 \uD569\uB2C8\uB2E4. \uB9C8\uC2A4\uD130 \uBE44\uBC00\uBC88\uD638\uB97C \uC785\uB825\uD558\uC5EC \uC774 \uC138\uC158\uC758 \uC7A0\uAE08\uC744 \uD574\uC81C\uD558\uC138\uC694. \uC774 \uAE30\uAE30\uC758 ${name}\uC5D0 \uC9C1\uC811 \uC804\uB2EC\uB418\uBA70 \uC5D0\uC774\uC804\uD2B8\uC5D0 \uC800\uC7A5\uB418\uAC70\uB098 \uD45C\uC2DC\uB418\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4.`,
    vaultUnlockTitle: name => `${name} \uC7A0\uAE08 \uD574\uC81C`
  },
  remoteDisplayBanner: {
    message: reason =>
      `\uC18C\uD504\uD2B8\uC6E8\uC5B4 \uB80C\uB354\uB9C1 \uD65C\uC131\uD654\uB428 \u2014 \uC6D0\uACA9 \uB514\uC2A4\uD50C\uB808\uC774 \uAC10\uC9C0\uB428 (${reason}). \uAE5C\uBC15\uC784\uC744 \uBC29\uC9C0\uD558\uAE30 \uC704\uD574 GPU \uAC00\uC18D\uC774 \uBE44\uD65C\uC131\uD654\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`
  },
  butterbar: {
    goTo: (index, total) => `공지 ${index}/${total} 표시`
  },
  rightSidebar: {
    couldNotPreview: path => `${path}\uC744(\uB97C) \uBBF8\uB9AC \uBCFC \uC218 \uC5C6\uC2B5\uB2C8\uB2E4`,
    folderTip: cwd => cwd,
    remotePickerCreateFolderFailed: error =>
      `\uD3F4\uB354\uB97C \uC0DD\uC131\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4 (${error}).`,
    unreadableBody: error => `\uC774 \uD3F4\uB354\uB97C \uC77D\uC744 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4 (${error}).`
  },
  sendDiagnostics: {
    uploadIdFallback: id =>
      `\uBC18\uD658\uB41C \uBCF4\uAE30 \uB9C1\uD06C\uAC00 \uC5C6\uC2B5\uB2C8\uB2E4 \u2014 \uC5C5\uB85C\uB4DC ID ${id}\uB97C \uC9C0\uC6D0\uD300\uC5D0 \uBB38\uC758\uD558\uC138\uC694`
  },
  settings: {
    appearance: {
      embedsReset: count => `\uD5C8\uC6A9\uB41C \uC11C\uBE44\uC2A4 ${count}\uAC1C \uC7AC\uC124\uC815`,
      installed: name => `\u201C${name}\u201D\uC744(\uB97C) \uC124\uCE58\uD588\uC2B5\uB2C8\uB2E4.`,
      pet: {
        adoptFailed: slug => `${slug}\uC744(\uB97C) \uC785\uC591\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4`,
        count: n => `\uD3AB ${n}\uB9C8\uB9AC.`,
        countCapped: (cap, total) =>
          `\uC804\uCCB4 ${total}\uAC1C \uC911 ${cap}\uAC1C \uD45C\uC2DC \uC911 \u2014 \uC881\uD600\uC11C \uAC80\uC0C9\uD558\uB824\uBA74 \uC785\uB825\uD558\uC138\uC694.`,
        delete: name => `${name} \uC0AD\uC81C`,
        deleteTitle: name => `${name}\uC744(\uB97C) \uC0AD\uC81C\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
        exportFailed: slug => `${slug}\uC744(\uB97C) \uB0B4\uBCF4\uB0BC \uC218 \uC5C6\uC2B5\uB2C8\uB2E4`,
        exportPet: name => `${name} \uB0B4\uBCF4\uB0B4\uAE30`,
        noMatch: query => `"${query}"\uC5D0 \uC77C\uCE58\uD558\uB294 \uD3AB\uC774 \uC5C6\uC2B5\uB2C8\uB2E4.`,
        rename: name => `${name} \uC774\uB984 \uBCC0\uACBD`,
        renameFailed: slug => `${slug}\uC758 \uC774\uB984\uC744 \uBCC0\uACBD\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4`,
        uninstall: name => `${name} \uC81C\uAC70`,
        uninstallFailed: slug => `${slug}\uC744(\uB97C) \uC81C\uAC70\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4`
      },
      themeProfileNote: profile =>
        `${profile} \uD504\uB85C\uD544\uC5D0 \uC800\uC7A5\uB428 \u2014 \uAC01 \uD504\uB85C\uD544\uC740 \uC790\uCCB4 \uD14C\uB9C8\uB97C \uC720\uC9C0\uD569\uB2C8\uB2E4.`,
      tipsReset: count => `\uD301 ${count}\uAC1C \uB2E4\uC2DC \uD45C\uC2DC`,
      uiScaleDesc: percent =>
        `\uC804\uCCB4 \uC571\uC758 \uD14D\uC2A4\uD2B8\uC640 \uCEE8\uD2B8\uB864 \uD06C\uAE30\uB97C \uC870\uC808\uD569\uB2C8\uB2E4. Cmd/Ctrl\uACFC +, -, 0 \uC870\uD569\uB3C4 \uC0AC\uC6A9\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4. \uD604\uC7AC: ${percent}%.`
    },
    billing: {
      amountValidation: {
        decimal: label =>
          `${label}: \uC18C\uC218\uC810 \uC774\uD558 \uCD5C\uB300 2\uC790\uB9AC\uAE4C\uC9C0 \uB2EC\uB7EC \uAE08\uC561\uC744 \uC785\uB825\uD558\uC138\uC694.`,
        maximum: (label, amount) => `${label}: \uCD5C\uB300\uAC12\uC740 ${amount}\uC785\uB2C8\uB2E4.`,
        minimum: (label, amount) => `${label}: \uCD5C\uC18C\uAC12\uC740 ${amount}\uC785\uB2C8\uB2E4.`,
        positive: label => `${label}: \uAE08\uC561\uC740 $0\uBCF4\uB2E4 \uCEE4\uC57C \uD569\uB2C8\uB2E4.`
      },
      buyCredits: {
        added: amount =>
          `${amount}\uC774(\uAC00) \uCD94\uAC00\uB418\uC5C8\uC2B5\uB2C8\uB2E4. \uC794\uC561\uC744 \uC0C8\uB85C \uACE0\uCE58\uB294 \uC911\uC785\uB2C8\uB2E4.`
      },
      charge: {
        added: amount =>
          amount
            ? `$${amount} \uCD94\uAC00\uB428.`
            : '\uD06C\uB808\uB527\uC774 \uCD94\uAC00\uB418\uC5C8\uC2B5\uB2C8\uB2E4.',
        failedBody: reason => `\uACB0\uC81C\uAC00 \uC2B9\uC778\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4 (${reason}).`,
        unconfirmedBody: message =>
          `${message} \uB9C8\uC9C0\uB9C9 \uACB0\uC81C \uACB0\uACFC\uB97C \uD655\uC778\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4. \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uAE30 \uC804\uC5D0 \uC794\uC561/\uB0B4\uC5ED\uC744 \uD655\uC778\uD558\uC138\uC694.`
      },
      creditsPerMonth: amount => `\uC6D4 ${amount} \uD06C\uB808\uB527`,
      errors: {
        monthlyCapExceeded: {
          messageHeadroom: remaining =>
            `\u{1F534} \uC6D4 \uC9C0\uCD9C \uD55C\uB3C4 \uB3C4\uB2EC \u2014 \uB0A8\uC740 \uC5EC\uC720 \uD55C\uB3C4 $${remaining}.`
        },
        rateLimited: {
          message: mins =>
            mins > 0
              ? `\u{1F7E1} \uD604\uC7AC \uC694\uCCAD\uC774 \uB108\uBB34 \uB9CE\uC2B5\uB2C8\uB2E4 (~${mins}\uBD84 \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694). \uACB0\uC81C \uC2E4\uD328\uAC00 \uC544\uB2D9\uB2C8\uB2E4.`
              : '\u{1F7E1} \uD604\uC7AC \uC694\uCCAD\uC774 \uB108\uBB34 \uB9CE\uC2B5\uB2C8\uB2E4. \uACB0\uC81C \uC2E4\uD328\uAC00 \uC544\uB2D9\uB2C8\uB2E4.'
        },
        remoteSpendingReconnect: who =>
          `${who} \uC774 \uAE30\uAE30\uB97C \uB2E4\uC2DC \uC2B9\uC778\uD558\uB824\uBA74 \uC124\uC815 -> Gateway\uC5D0\uC11C \uB2E4\uC2DC \uC5F0\uACB0\uD558\uC138\uC694.`,
        stripeUnavailable: {
          message: mins =>
            mins > 0
              ? `Stripe\uC5D0 \uBB38\uC81C\uAC00 \uBC1C\uC0DD\uD588\uC2B5\uB2C8\uB2E4 \u2014 ~${mins}\uBD84 \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694`
              : `Stripe\uC5D0 \uBB38\uC81C\uAC00 \uBC1C\uC0DD\uD588\uC2B5\uB2C8\uB2E4 \u2014 \uC7A0\uC2DC \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694`
        }
      },
      perMonth: amount => `\uC6D4 ${amount}`,
      plan: {
        alreadyOn: name =>
          `\uC774\uBBF8 ${name} \uD50C\uB79C\uC744 \uC0AC\uC6A9 \uC911\uC774\uBBC0\uB85C \uBCC0\uACBD\uD560 \uB0B4\uC6A9\uC774 \uC5C6\uC2B5\uB2C8\uB2E4.`,
        effectScheduled: (targetName, effectiveAt, creditsDelta) =>
          `${targetName}(\uC73C)\uB85C \uBCC0\uACBD \u2014 ${effectiveAt}\uC5D0 \uC801\uC6A9\uB429\uB2C8\uB2E4. \uC9C0\uAE08\uC740 \uCCAD\uAD6C\uB418\uC9C0 \uC54A\uC73C\uBA70, \uADF8\uB54C\uAE4C\uC9C0 \uD604\uC7AC \uD50C\uB79C\uC774 \uC720\uC9C0\uB429\uB2C8\uB2E4.${creditsDelta ? ` \uC6D4\uAC04 \uD06C\uB808\uB527 \uBCC0\uACBD: ${creditsDelta}.` : ''}`
      },
      state: {
        autoRefill: {
          chargesDescription: (reloadTo, threshold) =>
            `\uC794\uC561\uC774 ${threshold} \uC544\uB798\uB85C \uB5A8\uC5B4\uC9C0\uBA74 ${reloadTo}(\uC774)\uAC00 \uC790\uB3D9\uC73C\uB85C \uCDA9\uC804\uB429\uB2C8\uB2E4.`,
          distinctCardCaption: cardLabel =>
            `\uC790\uB3D9 \uCDA9\uC804 ${cardLabel} \uACB0\uC81C \u2014 \uD3EC\uD138\uC5D0\uC11C \uB0B4\uC5ED \uD655\uC778`
        },
        paymentMethod: {
          provenance: {
            suffix: label => ` - ${label}`
          }
        },
        planCard: {
          cancellationCaption: when => `${when}\uC5D0 \uCDE8\uC18C\uB429\uB2C8\uB2E4.`,
          downgradeCaption: (tierName, when) =>
            `${when}\uC5D0 ${tierName}(\uC73C)\uB85C \uBCC0\uACBD\uB429\uB2C8\uB2E4.`,
          renewsCaption: date => `${date} \uAC31\uC2E0`
        },
        usage: {
          monthlyCap: {
            valueUsed: (spent, limit) => `${limit} \uC911 ${spent} \uC0AC\uC6A9\uB428`
          },
          subscriptionCredits: {
            title: '구독 크레딧',
            barLabel: '남은 구독 크레딧',
            captionResets: date => `${date} \uCD08\uAE30\uD654`,
            valueOf: (remaining, monthly) => `${monthly} \uC911 ${remaining} \uB0A8\uC74C`,
            valueOver: (remaining, monthly, over) =>
              `${monthly} \uC911 ${remaining} \uB0A8\uC74C \xB7 ${over} \uCD08\uACFC`
          }
        }
      },
      usageLabel: label => `${label} \uC0AC\uC6A9\uB7C9`
    },
    connections: {
      duplicateSsh: label =>
        `\uC774 SSH \uD638\uC2A4\uD2B8\uC5D0 \uB300\uD55C \uC5F0\uACB0\uC774 \uC774\uBBF8 \uC874\uC7AC\uD569\uB2C8\uB2E4 (\u201C${label}\u201D).`,
      duplicateUrl: label =>
        `\uC774 \uAC8C\uC774\uD2B8\uC6E8\uC774 URL\uC5D0 \uB300\uD55C \uC5F0\uACB0\uC774 \uC774\uBBF8 \uC874\uC7AC\uD569\uB2C8\uB2E4 (\u201C${label}\u201D).`,
      removeConfirmDesc: label =>
        `\u201C${label}\u201D\uC774(\uAC00) \uC774 \uC571\uC5D0\uC11C \uC81C\uAC70\uB429\uB2C8\uB2E4. \uC778\uC2A4\uD134\uC2A4 \uC790\uCCB4\uB294 \uC0AD\uC81C\uB418\uC9C0 \uC54A\uC73C\uBBC0\uB85C \uC5B8\uC81C\uB4E0\uC9C0 \uB2E4\uC2DC \uCD94\uAC00\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4.`,
      sameBackendHint: label => `\u201C${label}\u201D\uACFC(\uC640) \uB3D9\uC77C\uD55C \uBC31\uC5D4\uB4DC`
    },
    credentials: {
      pasteLabelKey: label => `${label} \uD0A4 \uBD99\uC5EC\uB123\uAE30`
    },
    customEndpoints: {
      deleteConfirm: name => `${name}\uC744(\uB97C) \uC0AD\uC81C\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
      endpointReachableModels: (reachable, count) => `${reachable} \uBAA8\uB378 ${count}\uAC1C \uBC1C\uACAC\uB428.`,
      endpointReachableTransport: transport =>
        `\uC5D4\uB4DC\uD3EC\uC778\uD2B8\uC5D0 \uB3C4\uB2EC\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4 (${transport} \uB77C\uC6B0\uD2B8 \uC81C\uACF5\uB428).`
    },
    gateway: {
      authNeedsOauth: provider =>
        `\uC774 \uAC8C\uC774\uD2B8\uC6E8\uC774\uB294 OAuth\uB97C \uC0AC\uC6A9\uD569\uB2C8\uB2E4. ${provider}(\uC73C)\uB85C \uB85C\uADF8\uC778\uD558\uC5EC \uC774 \uB370\uC2A4\uD06C\uD1B1 \uC571\uC744 \uC2B9\uC778\uD558\uC138\uC694.`,
      cloudConnectedTo: name => `${name}\uC5D0 \uC5F0\uACB0\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      cloudOrgRole: role => `\uC5ED\uD560: ${role}`,
      cloudStatusLabel: status => `\uC0C1\uD0DC: ${status}`,
      connectedTo: (baseUrl, version) =>
        `${baseUrl}${version ? ` \xB7 Hermes ${version}` : ''}\uC5D0 \uC5F0\uACB0\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      existingToken: value => `\uAE30\uC874 \uD1A0\uD070 ${value}`,
      signInWith: provider => `${provider}(\uC73C)\uB85C \uB85C\uADF8\uC778`,
      sshReachable: (host, platform) =>
        `\uC5F0\uACB0 \uAC00\uB2A5: ${host} (${platform}) \u2014 Hermes \uAC10\uC9C0\uB428`
    },
    localModels: {
      activateDoneToast: model =>
        `\uC0C8 \uB300\uD654\uC5D0\uC11C ${model}\uC744(\uB97C) \uC0AC\uC6A9\uD569\uB2C8\uB2E4.`,
      activateFailed: model => `${model}(\uC73C)\uB85C \uC804\uD658\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      deleteConfirm: model =>
        `\uB514\uC2A4\uD06C\uC5D0\uC11C ${model}\uC744(\uB97C) \uC0AD\uC81C\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
      deleted: model => `${model}\uC774(\uAC00) \uC0AD\uC81C\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      downloadAction: size => `\uB2E4\uC6B4\uB85C\uB4DC \xB7 ${size}`,
      downloadDoneToast: model => `${model} \uC900\uBE44\uAC00 \uC644\uB8CC\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      downloadEta: time => `\uC57D ${time} \uB0A8\uC74C`,
      downloadEtaHours: (hours, minutes) =>
        minutes ? `${hours}\uC2DC\uAC04 ${minutes}\uBD84` : `${hours}\uC2DC\uAC04`,
      downloadEtaMinutes: count => `${count}\uBD84`,
      downloadEtaSeconds: count => `${count}\uCD08`,
      downloadFailed: model => `${model} \uB2E4\uC6B4\uB85C\uB4DC \uC2E4\uD328`,
      downloadPauseFailed: model =>
        `${model} \uB2E4\uC6B4\uB85C\uB4DC\uB97C \uC77C\uC2DC \uC911\uC9C0\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      downloadProgress: (done, total) => `${total} \uC911 ${done}`,
      downloadResumeFailed: model =>
        `${model} \uB2E4\uC6B4\uB85C\uB4DC\uB97C \uC7AC\uAC1C\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      downloadSpeed: rate => `${rate}`,
      pillFullContext: max => `\uC804\uCCB4 ${max} \uCEE8\uD14D\uC2A4\uD2B8`,
      pillUpTo: max => `\uCD5C\uB300 ${max} \uCEE8\uD14D\uC2A4\uD2B8`,
      quickstartDetail: (model, size) =>
        `\uD074\uB9AD \uD55C \uBC88\uC73C\uB85C \uB85C\uCEEC \uC5D4\uC9C4, ${model} (${size} \uB2E4\uC6B4\uB85C\uB4DC), \uC0C8 \uB300\uD654 \uAE30\uBCF8 \uC124\uC815\uC774 \uBAA8\uB450 \uC644\uB8CC\uB429\uB2C8\uB2E4. \uB370\uC774\uD130\uB294 \uC774 \uCEF4\uD4E8\uD130 \uC678\uBD80\uB85C \uC804\uC1A1\uB418\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4.`,
      quickstartDetailReady: model =>
        `\uD074\uB9AD \uD55C \uBC88\uC73C\uB85C \uC0C8 \uB300\uD654\uC758 \uAE30\uBCF8 \uBAA8\uB378\uC774 ${model}(\uC73C)\uB85C \uC124\uC815\uB429\uB2C8\uB2E4. \uBAA8\uB4E0 \uC791\uC5C5\uC774 \uC774 \uAE30\uAE30\uC5D0\uC11C \uC2E4\uD589\uB429\uB2C8\uB2E4.`,
      quickstartDoneToast: model =>
        `${model} \uC124\uC815\uC774 \uC644\uB8CC\uB418\uC5C8\uC2B5\uB2C8\uB2E4 \u2014 \uC0C8 \uB300\uD654\uAC00 \uC774 \uAE30\uAE30\uC5D0\uC11C \uC2E4\uD589\uB429\uB2C8\uB2E4.`,
      ram: label => `${label} RAM`,
      runtimeInstalledDetail: (tag, backend) =>
        `\uBE4C\uB4DC ${tag}, ${backend} \uBC31\uC5D4\uB4DC. Hermes\uAC00 \uC11C\uBC84\uB97C \uC2DC\uC791\uD558\uACE0 \uAD00\uB9AC\uD569\uB2C8\uB2E4.`,
      runtimeReady: backend => `\uC900\uBE44\uB428 \xB7 ${backend}`,
      upToDateDetail: (tag, backend) => `llama.cpp ${tag} (${backend}) \uC2E4\uD589 \uC911.`,
      updateDetail: (next, current) =>
        `\uCD5C\uC2E0 llama.cpp \uBE4C\uB4DC(${next})\uB97C \uC124\uCE58\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4 \u2014 \uD604\uC7AC \uBC84\uC804: ${current}. \uB2E4\uC6B4\uB85C\uB4DC \uC911\uC5D0\uB3C4 \uBAA8\uB378\uC740 \uACC4\uC18D \uC791\uB3D9\uD569\uB2C8\uB2E4.`,
      vram: label => `${label} GPU \uBA54\uBAA8\uB9AC`
    },
    managedUpdates: {
      receipt: (id, outcome) => `\uC218\uC2E0\uC99D ${id} \xB7 ${outcome}`,
      receiptVersions: (pre, post) => `${pre} \u2192 ${post}`,
      scopeNotRestored: (profile, error) =>
        `\u201C${profile}\u201D \uD504\uB85C\uD544\uC744 \uBCF5\uC6D0\uD558\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4: ${error}`,
      scopesRestored: profiles => `\uBCF5\uC6D0\uB41C \uD504\uB85C\uD544: ${profiles}`
    },
    mcp: {
      authenticatedMessage: (server, count) => `${server}: \uB3C4\uAD6C ${count}\uAC1C`,
      capabilitySummary: (tools, prompts, resources) =>
        `${[`\uB3C4\uAD6C ${tools}\uAC1C`, ...(prompts ? [`\uD504\uB86C\uD504\uD2B8 ${prompts}\uAC1C`] : []), ...(resources ? [`\uB9AC\uC18C\uC2A4 ${resources}\uAC1C`] : [])].join(', ')} \uD65C\uC131\uD654\uB428`,
      catalogInstallFailed: name => `${name} \uC124\uCE58\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
      costTokens: tokens => `\uC57D ${tokens} \uD1A0\uD070/\uD638\uCD9C`,
      deepLinkNameConflict: name =>
        `\u201C${name}\u201D(\uC774)\uB77C\uB294 \uC774\uB984\uC758 \uC11C\uBC84\uAC00 \uC774\uBBF8 \uC874\uC7AC\uD569\uB2C8\uB2E4 \u2014 \uB2E4\uB978 \uC774\uB984\uC744 \uC120\uD0DD\uD558\uAC70\uB098 \uCDE8\uC18C\uD558\uC138\uC694.`,
      savedMessage: name =>
        `MCP\uB97C \uB2E4\uC2DC \uB85C\uB4DC\uD558\uBA74 ${name}\uC774(\uAC00) \uC801\uC6A9\uB429\uB2C8\uB2E4.`,
      usage30d: uses => `30\uC77C\uAC04 ${uses}\uD68C \uC0AC\uC6A9`
    },
    model: {
      mainAppliedMessage: model =>
        `\uC0C8 \uC138\uC158\uC5D0\uC11C ${model}\uC744(\uB97C) \uC0AC\uC6A9\uD569\uB2C8\uB2E4.`,
      inheritsFrom: task => `${task}에서 상속`,
      followTask: task => `${task} 따라가기`,
      moaReferenceTitle: index => `\uCC38\uC870 ${index}`,
      moaReferenceToggle: (enabled, index) =>
        `\uCC38\uC870 ${index} ${enabled ? '\uC0AC\uC6A9 \uC548 \uD568' : '\uC0AC\uC6A9'}`,
      setUpProvider: name => `${name} \uC124\uC815`,
      staleAuxBefore: (count, names) =>
        `${names}\uC758 \uBCF4\uC870 \uC791\uC5C5 ${count}\uAC1C\uAC00 \uC5EC\uC804\uD788 \uB2E4\uC74C\uC5D0\uC11C \uC2E4\uD589 \uC911\uC785\uB2C8\uB2E4: `
    },
    plugins: {
      installModal: {
        agentSuccess: name =>
          `\uC5D0\uC774\uC804\uD2B8 \uD50C\uB7EC\uADF8\uC778 ${name}\uC774(\uAC00) \uC124\uCE58\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
        agentTargetLocal: (profile, dir) =>
          `${profile} \uBC31\uC5D4\uB4DC(${dir})\uC5D0 \uC124\uCE58\uD569\uB2C8\uB2E4.`,
        agentTargetRemote: profile =>
          `\uC5F0\uACB0\uB41C ${profile} \uBC31\uC5D4\uB4DC\uC5D0 \uC124\uCE58\uD569\uB2C8\uB2E4.`,
        alreadyInstalled: name =>
          `${name}\uC774(\uAC00) \uC774\uBBF8 \uC124\uCE58\uB418\uC5B4 \uC788\uC2B5\uB2C8\uB2E4.`,
        catalogPinned: (name, sha) =>
          `Hermes \uCE74\uD0C8\uB85C\uADF8 \uD56D\uBAA9 \u201C${name}\u201D \u2014 \uC5D0\uC774\uC804\uD2B8 \uAD6C\uC131 \uC694\uC18C\uB294 \uBE0C\uB79C\uCE58 \uD301\uC774 \uC544\uB2C8\uB77C \uAC80\uD1A0\uB41C \uACE0\uC815 \uBC84\uC804${sha ? ` ${sha}` : ''}(\uC73C)\uB85C \uC124\uCE58\uB429\uB2C8\uB2E4.`,
        desktopSuccess: name =>
          `\uB370\uC2A4\uD06C\uD1B1 \uD50C\uB7EC\uADF8\uC778 ${name}\uC774(\uAC00) \uC124\uCE58\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
        missingEnv: (name, vars) =>
          `${name}\uC774(\uAC00) \uC124\uCE58\uB418\uC5C8\uC9C0\uB9CC \uC791\uB3D9\uD558\uB824\uBA74 \uB2E4\uC74C \uD0A4\uAC00 \uD544\uC694\uD569\uB2C8\uB2E4: ${vars}. \uC9C0\uAE08 \uCD94\uAC00\uD558\uC9C0 \uC54A\uC73C\uBA74 \uD50C\uB7EC\uADF8\uC778\uC758 \uB3C4\uAD6C\uAC00 \uC791\uB3D9\uD558\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4.`,
        serverNotConnected: (server, reason) =>
          `MCP \uC11C\uBC84 ${server}\uC5D0 \uC5F0\uACB0\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4${reason ? `: ${reason}` : '.'}`,
        skillsReady: names =>
          names.length === 1
            ? `\uC2A4\uD0AC ${names[0]}\uC900\uBE44\uB428`
            : `\uC2A4\uD0AC ${names.length}\uAC1C \uC900\uBE44\uB428`,
        toolsConnected: n => `\uC5F0\uACB0\uB41C \uB3C4\uAD6C ${n}\uAC1C`
      }
    },
    profileScope: {
      editsProfile: profile =>
        `\uC774 \uD398\uC774\uC9C0\uC758 \uBCC0\uACBD \uC0AC\uD56D\uC740 \u201C${profile}\u201D \uD504\uB85C\uD544\uC5D0 \uC801\uC6A9\uB429\uB2C8\uB2E4.`
    },
    providers: {
      failedRemove: provider => `${provider}\uC744(\uB97C) \uC81C\uAC70\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      removeConfirm: provider => `${provider}\uC744(\uB97C) \uC81C\uAC70\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
      removeExternalGeneric: provider =>
        `${provider}\uC740(\uB294) \uC790\uCCB4 CLI\uB85C \uAD00\uB9AC\uB429\uB2C8\uB2E4 \u2014 \uD574\uB2F9 CLI\uC5D0\uC11C \uC81C\uAC70\uD558\uC138\uC694.`,
      removeKeyManaged: provider =>
        `${provider}\uC740(\uB294) API \uD0A4\uB85C \uAD6C\uC131\uB429\uB2C8\uB2E4. API \uD0A4\uC5D0\uC11C \uC81C\uAC70\uD558\uC138\uC694.`,
      removeTerminalConfirm: (provider, command) =>
        `${provider} \uC5F0\uACB0\uC744 \uD574\uC81C\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C? \uD130\uBBF8\uB110\uC5D0\uC11C \u201C${command}\u201D \uBA85\uB839\uC744 \uC2E4\uD589\uD558\uC5EC \uC790\uACA9 \uC99D\uBA85\uC744 \uC9C0\uC6C1\uB2C8\uB2E4.`,
      removeTerminalRunning: provider =>
        `\uD130\uBBF8\uB110\uC5D0\uC11C ${provider} \uC5F0\uACB0 \uD574\uC81C \uC911\u2026`,
      removedMessage: provider => `${provider}\uC774(\uAC00) \uC81C\uAC70\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`
    },
    sessions: {
      defaultsTo: label => `\uAE30\uBCF8\uAC12: ${label}.`,
      deleteConfirm: title =>
        `\u201C${title}\u201D\uC744(\uB97C) \uC601\uAD6C\uC801\uC73C\uB85C \uC0AD\uC81C\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C? \uC774 \uC791\uC5C5\uC740 \uCDE8\uC18C\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      messages: count => `\uBA54\uC2DC\uC9C0 ${count}\uAC1C`
    },
    toolsets: {
      failedRemove: key => `${key} \uC81C\uAC70\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
      failedReveal: key => `${key} \uD45C\uC2DC\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
      failedSave: key => `${key} \uC800\uC7A5\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
      failedSelect: provider => `${provider} \uC120\uD0DD\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
      failedSelectCapability: provider => `${provider} \uC124\uC815\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
      failedSelectModel: model => `${model} \uC120\uD0DD\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
      modelCount: count => `\uBAA8\uB378 ${count}\uAC1C`,
      modelSelectedMessage: model => `${model}\uC774(\uAC00) \uC0C8 \uC138\uC158\uC5D0 \uC801\uC6A9\uB429\uB2C8\uB2E4.`,
      nousAuthNeededMessage: provider =>
        `${provider}\uC774(\uAC00) \uC800\uC7A5\uB418\uC5C8\uC9C0\uB9CC Nous \uACC4\uC815\uC73C\uB85C \uB85C\uADF8\uC778\uD574\uC57C\uB9CC \uC791\uB3D9\uD569\uB2C8\uB2E4.`,
      postSetupCompleteMessage: step => `${step} \uC124\uCE58\uAC00 \uC644\uB8CC\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      postSetupErrorMessage: step =>
        `${step} \uC124\uC815\uC774 \uC644\uB8CC\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. \uB85C\uADF8\uB97C \uD655\uC778\uD558\uC5EC \uC6D0\uC778\uC744 \uD30C\uC545\uD55C \uD6C4 \uC124\uC815\uC744 \uB2E4\uC2DC \uC2E4\uD589\uD558\uC138\uC694.`,
      postSetupFailed: step => `${step} \uC124\uC815 \uC2E4\uD589\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
      postSetupHint: step =>
        `\uC774 \uBC31\uC5D4\uB4DC\uB294 \uC77C\uD68C\uC131 \uC124\uCE58\uAC00 \uD544\uC694\uD569\uB2C8\uB2E4 (${step}). \uC774 \uAE30\uAE30\uC5D0\uC11C \uC2E4\uD589\uB418\uBA70 \uBA87 \uBD84 \uC815\uB3C4 \uAC78\uB9B4 \uC218 \uC788\uC2B5\uB2C8\uB2E4.`,
      removeConfirm: key => `.env\uC5D0\uC11C ${key}\uC744(\uB97C) \uC81C\uAC70\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
      removedMessage: key => `${key}\uC774(\uAC00) \uC81C\uAC70\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      savedMessage: key => `${key}\uC774(\uAC00) \uC5C5\uB370\uC774\uD2B8\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      selectedMessage: provider => `\uC774\uC81C ${provider}\uC774(\uAC00) \uD65C\uC131\uD654\uB429\uB2C8\uB2E4.`,
      terminalBackend: {
        failedSelect: backend => `${backend} \uC120\uD0DD\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
        needsSetupConfirmDescription: detail =>
          `${detail} \uC774 \uBCC0\uACBD \uC774\uD6C4 \uC2DC\uC791\uB418\uB294 \uC138\uC158\uC740 \uC124\uC815\uC774 \uC644\uB8CC\uB420 \uB54C\uAE4C\uC9C0 \uD130\uBBF8\uB110\uC774\uB098 \uD30C\uC77C \uB3C4\uAD6C\uB97C \uC0AC\uC6A9\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
        needsSetupConfirmTitle: backend =>
          `\uADF8\uB798\uB3C4 ${backend}\uC744(\uB97C) \uC120\uD0DD\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
        selectedMessage: backend =>
          `\uC774\uC81C \uD130\uBBF8\uB110 \uBA85\uB839\uC774 ${backend}\uC744(\uB97C) \uD1B5\uD574 \uC2E4\uD589\uB429\uB2C8\uB2E4. \uC0C8 \uC138\uC158\uC5D0 \uC801\uC6A9\uB429\uB2C8\uB2E4.`,
        unavailableMessage: backend =>
          `\uD604\uC7AC Hermes\uAC00 \uC258 \uBA85\uB839\uC744 \uC2E4\uD589\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4. ${backend}\uC774(\uAC00) \uC900\uBE44\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. \uB85C\uCEEC\uB85C \uC804\uD658\uD558\uAC70\uB098 ${backend} \uC124\uC815\uC744 \uC644\uB8CC\uD55C \uD6C4 \uB2E4\uC2DC \uC2DC\uB3C4\uD558\uC138\uC694.`
      },
      webCapabilitySelectedMessage: (provider, capability) =>
        `\uC774\uC81C ${provider}\uC774(\uAC00) \uC6F9 ${capability}\uC744(\uB97C) \uCC98\uB9AC\uD569\uB2C8\uB2E4.`,
      webExtractActive: backend => `\uCD94\uCD9C: ${backend}`,
      webSearchActive: backend => `\uAC80\uC0C9: ${backend}`
    },
    uninstallSection: {
      confirmBody: what =>
        `${what}\uC744(\uB97C) \uC81C\uAC70\uD569\uB2C8\uB2E4. \uC774 \uC791\uC5C5\uC740 \uB418\uB3CC\uB9B4 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      dataKept: path => `설정, 채팅, 비밀은 ${path}에 있습니다. 앱을 삭제해도 삭제되지 않습니다.`
    },
    vault: {
      count: n => `${n}\uAC1C \uC800\uC7A5\uB428`,
      createdOn: date => `${date} \uCD94\uAC00\uB428`,
      deleteDescription: label =>
        `"${label}"\uC774(\uAC00) \uC81C\uAC70\uB429\uB2C8\uB2E4. \uC774 \uC791\uC5C5\uC740 \uB418\uB3CC\uB9B4 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      identifierShown: identifier => identifier,
      sources: {
        notInstalled: name =>
          `\uAC10\uC9C0\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. ${name} \uBA85\uB839\uC904 \uB3C4\uAD6C\uB97C \uC124\uCE58\uD558\uACE0 \uB85C\uADF8\uC778\uD558\uBA74 Hermes\uAC00 \uC790\uB3D9\uC73C\uB85C \uC778\uC2DD\uD569\uB2C8\uB2E4.`,
        unlockTitle: name => `${name} \uC7A0\uAE08 \uD574\uC81C`,
        unlocked: name =>
          `\uC774 \uC138\uC158 \uB3D9\uC548 ${name} \uC7A0\uAE08\uC774 \uD574\uC81C\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`
      }
    },
    pluginPages: {
      pageCount: n => (n === 1 ? '1개 페이지' : `${n}개 페이지`)
    }
  },
  shell: {
    approvalMode: {
      ariaLabel: mode => `\uC2B9\uC778 \uBAA8\uB4DC: ${mode}`
    },
    gatewayMenu: {
      connection: label => `\uC5F0\uACB0: ${label}`
    },
    modelMenu: {
      priceTitle: (input, output, cache) =>
        `\uC785\uB825 ${input}/Mtok \xB7 \uCD9C\uB825 ${output}/Mtok` +
        (cache ? ` \xB7 \uCE90\uC2DC\uB41C \uC77D\uAE30 ${cache}/Mtok` : ''),
      limitedUntil: time => `${time}까지 제한됨`,
      limitedTip: (provider, time) =>
        time
          ? `${provider}의 사용 한도에 도달했습니다. ${time}에 초기화되며, 이후에 사용할 모델은 지금 선택할 수 있습니다.`
          : `${provider}의 사용 한도에 도달했습니다. 초기화 이후에 사용할 모델은 지금 선택할 수 있습니다.`,
      modelResets: time => `${time} 초기화`,
      modelLimitedTip: time =>
        `이 모델은 자체 한도에 도달해 ${time}에 초기화됩니다. 여기의 다른 모델은 계속 사용할 수 있습니다.`,
      usageLeft: (percent, time) => (time ? `${percent}% 남음 · ${time} 초기화` : `${percent}% 남음`),
      poolAccounts: count => `${count}개 계정`,
      poolLimited: (limited, total) => `${limited}/${total}개 계정 제한`,
      poolAccount: number => `계정 ${number}`,
      usageTip: provider => `${provider}: 사용 한도가 얼마 남지 않았습니다.`,
      usageWindow: (label, percent, time) =>
        time ? `${label}: ${percent}% 남음, ${time} 초기화` : `${label}: ${percent}% 남음`
    },
    modelOptions: {
      sendsOnRoute: level => `\uC774 \uACBD\uB85C\uC5D0\uC11C ${level} \uC804\uC1A1`
    },
    statusbar: {
      backendLabel: version => `\uBC31\uC5D4\uB4DC v${version}`,
      backendVersion: version => `\uBC31\uC5D4\uB4DC v${version}`,
      branch: branch => `\uBE0C\uB79C\uCE58 ${branch}`,
      messagingDegraded: name => `${name} 다운`,
      clientLabel: version => `\uD074\uB77C\uC774\uC5B8\uD2B8 v${version}`,
      commit: sha => `\uCEE4\uBC0B ${sha}`,
      commitsBehind: (count, branch) => `${branch}\uBCF4\uB2E4 ${count}\uAC1C \uCEE4\uBC0B \uB4A4\uCC98\uC9D0`,
      compressions: count => `\uC555\uCD95: ${count}`,
      connectionCloud: host => `\uD074\uB77C\uC6B0\uB4DC: ${host}`,
      connectionCloudTooltip: host => `Hermes \uD074\uB77C\uC6B0\uB4DC \xB7 ${host}`,
      connectionRemote: host => `\uC6D0\uACA9: ${host}`,
      connectionRemoteTooltip: host => `\uC6D0\uACA9 \xB7 ${host}`,
      connectionSsh: host => `SSH: ${host}`,
      connectionSshTooltip: host => `SSH \xB7 ${host}`,
      contextUsagePanel: {
        percentFull: percent => `${percent}% \uC0AC\uC6A9\uB428`,
        tokenSummary: (used, max) => `${used} / ${max} \uD1A0\uD070`
      },
      desktopVersion: version => `Hermes \uB370\uC2A4\uD06C\uD1B1 v${version}`,
      failed: count => `${count}\uAC1C \uC2E4\uD328`,
      modelTitle: (provider, model) => `\uBAA8\uB378 \xB7 ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} \xB7 ${model}`,
      releaseAvailable: tag => `\uBC84\uC804 ${tag}\uC744(\uB97C) \uC0AC\uC6A9\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4.`,
      running: count => `${count}\uAC1C \uC2E4\uD589 \uC911`,
      subagents: count => `\uC11C\uBE0C\uC5D0\uC774\uC804\uD2B8 ${count}\uAC1C`
    }
  },
  skillDeepLink: {
    installComplete: name => `“${name}”이(가) 설치되었습니다.`,
    installTitle: name => `“${name}”을(를) 설치하시겠습니까?`
  },
  skills: {
    appliesToNewSessions: name => `${name}\uC740(\uB294) \uC0C8 \uC138\uC158\uC5D0 \uC801\uC6A9\uB429\uB2C8\uB2E4.`,
    bulkUpdated: count =>
      `\uC0C8 \uC138\uC158\uC744 \uC704\uD574 ${count}\uAC1C \uD56D\uBAA9\uC774 \uC5C5\uB370\uC774\uD2B8\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
    configureToolset: label => `${label} \uC124\uC815`,
    emptyNoneAvailable: noun => `\uC0AC\uC6A9 \uAC00\uB2A5\uD55C ${noun}\uC774(\uAC00) \uC5C6\uC2B5\uB2C8\uB2E4.`,
    emptyNoneFound: noun => `${noun}\uC744(\uB97C) \uCC3E\uC744 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
    emptyNothingMatches: query =>
      `\u201C${query}\u201D\uACFC(\uC640) \uC77C\uCE58\uD558\uB294 \uD56D\uBAA9\uC774 \uC5C6\uC2B5\uB2C8\uB2E4.`,
    failedToUpdate: name => `${name} \uC5C5\uB370\uC774\uD2B8\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.`,
    hub: {
      alreadyInstalled: name =>
        `"${name}"\uC774(\uAC00) \uC774\uBBF8 \uC124\uCE58\uB418\uC5B4 \uC788\uC2B5\uB2C8\uB2E4.`,
      findings: count => `\uBC1C\uACAC\uB41C \uD56D\uBAA9 ${count}\uAC1C`,
      installBlockedMessage: (findings, unverified) =>
        `\uBCF4\uC548 \uAC80\uC0AC\uC5D0\uC11C \uAC80\uD1A0\uD560 ${findings > 0 ? `${findings}\uAC1C \uD56D\uBAA9` : '\uC704\uD5D8\uD55C \uD328\uD134'}\uC744(\uB97C) \uAC10\uC9C0\uD588\uC2B5\uB2C8\uB2E4${unverified ? ' (\uC774 \uC2A4\uD0AC\uC740 \uD655\uC778\uB418\uC9C0 \uC54A\uC740 \uCD9C\uCC98\uC5D0\uC11C \uC654\uC2B5\uB2C8\uB2E4)' : ''}. \uC791\uC131\uC790\uB97C \uC2E0\uB8B0\uD560\uC9C0 \uACB0\uC815\uD558\uAE30 \uC804\uC5D0 \uAC80\uC0AC \uACB0\uACFC\uB97C \uC77D\uC5B4\uBCF4\uC138\uC694.`,
      installBlockedTitle: name => `${name}\uC744(\uB97C) \uC124\uCE58\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      installStarted: name => `${name} \uC124\uCE58 \uC911...`,
      resultCount: (count, ms) => `${count}\uAC1C \uACB0\uACFC${ms !== null ? ` (${ms}ms \uC18C\uC694)` : ''}`,
      timedOut: sources => `\uC2DC\uAC04 \uCD08\uACFC: ${sources}`,
      uninstallStarted: name => `${name} \uC81C\uAC70 \uC911...`
    },
    plugins: {
      alreadyInstalled: name =>
        `${name}\uC774(\uAC00) \uC774 \uD504\uB85C\uD544\uC5D0 \uC774\uBBF8 \uC124\uCE58\uB418\uC5B4 \uC788\uC2B5\uB2C8\uB2E4.`,
      catalogProvenance: sha =>
        `Hermes \uCE74\uD0C8\uB85C\uADF8\uC5D0\uC11C \uC124\uCE58\uB418\uC5C8\uC2B5\uB2C8\uB2E4${sha ? ` (\uD540 ${sha})` : ''}.`,
      deepLinkCatalogUnknown: name =>
        `\u201C${name}\u201D\uC740(\uB294) Hermes \uD50C\uB7EC\uADF8\uC778 \uCE74\uD0C8\uB85C\uADF8\uC5D0 \uC5C6\uC2B5\uB2C8\uB2E4. \uC124\uCE58\uB41C \uD56D\uBAA9\uC774 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      halfAgentIn: profile => `${profile}\uC758 \uC5D0\uC774\uC804\uD2B8`,
      installAgentHereTip: profile =>
        `\uB370\uC2A4\uD06C\uD1B1 \uD30C\uD2B8\uB294 \uC774 \uC571\uC5D0 \uB85C\uB4DC\uB418\uC5C8\uC9C0\uB9CC, \uC5D0\uC774\uC804\uD2B8 \uD30C\uD2B8\uB294 ${profile}\uC5D0 \uC124\uCE58\uB418\uC5B4 \uC788\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4. \uAC70\uAE30\uC11C \uC124\uCE58\uD558\uC138\uC694.`,
      pinnedBadge: sha => `\uD540 \uACE0\uC815\uB428 @ ${sha}`,
      pinnedProvenance: sha =>
        `\uCEE4\uBC0B ${sha}\uC5D0 \uACE0\uC815\uB418\uC5C8\uC2B5\uB2C8\uB2E4. \uC0C8 \uD540\uC73C\uB85C \uB2E4\uC2DC \uC124\uCE58\uD560 \uB54C\uAE4C\uC9C0 \uC5C5\uB370\uC774\uD2B8\uAC00 \uAC70\uBD80\uB429\uB2C8\uB2E4.`,
      settingsForm: {
        saveFailed: name => `${name} \uC124\uC815\uC744 \uC800\uC7A5\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
        saved: name => `${name} \uC124\uC815\uC774 \uC800\uC7A5\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
        secretStoredAs: env =>
          `config.yaml\uC774 \uC544\uB2CC \uD504\uB85C\uD544\uC758 .env\uC5D0 ${env}(\uC73C)\uB85C \uC800\uC7A5\uB429\uB2C8\uB2E4. \uD604\uC7AC \uAC12\uC744 \uC720\uC9C0\uD558\uB824\uBA74 \uBE44\uC6CC \uB450\uC138\uC694.`
      },
      settingsToggle: name => `\uC124\uC815: ${name}`,
      toggleFailed: name => `${name}\uC744(\uB97C) \uC804\uD658\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4.`,
      toolsetOff: (name, profile) =>
        `${profile}\uC5D0 \uB300\uD55C ${name} \uC5D0\uC774\uC804\uD2B8 \uB3C4\uAD6C\uAC00 \uBE44\uD65C\uC131\uD654\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      toolsetOn: (name, profile) =>
        `${profile}\uC5D0 \uB300\uD55C ${name} \uC5D0\uC774\uC804\uD2B8 \uB3C4\uAD6C\uAC00 \uD65C\uC131\uD654\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      toolsetToggleFailed: name =>
        `${name} \uC5D0\uC774\uC804\uD2B8 \uB3C4\uAD6C\uB97C \uC804\uD658\uD560 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4. \uB370\uC2A4\uD06C\uD1B1 \uD328\uB110\uC740 \uBCC0\uACBD\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4.`,
      uninstallConfirmBody: (name, profile) =>
        `${profile} \uD504\uB85C\uD544\uC5D0\uC11C \uD50C\uB7EC\uADF8\uC778 \uD30C\uC77C\uC744 \uC0AD\uC81C\uD569\uB2C8\uB2E4. \uD568\uAED8 \uC81C\uACF5\uB41C \uB370\uC2A4\uD06C\uD1B1 \uAE30\uB2A5\uB3C4 \uD568\uAED8 \uC81C\uAC70\uB429\uB2C8\uB2E4. \uCE74\uD0C8\uB85C\uADF8\uB098 Git\uC5D0\uC11C \uC5B8\uC81C\uB4E0\uC9C0 \uB2E4\uC2DC \uC124\uCE58\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4.`,
      uninstallConfirmTitle: name => `${name}\uC744(\uB97C) \uC81C\uAC70\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
      uninstallDesktopConfirmBody: name =>
        `\uC774 \uCEF4\uD4E8\uD130\uC758 desktop-plugins \uD3F4\uB354\uC5D0\uC11C ${name}\uC744(\uB97C) \uC0AD\uC81C\uD558\uACE0 \uC989\uC2DC \uC5B8\uB85C\uB4DC\uD569\uB2C8\uB2E4. Git\uC5D0\uC11C \uB2E4\uC2DC \uC124\uCE58\uD558\uAC70\uB098 \uC5B8\uC81C\uB4E0\uC9C0 \uD3F4\uB354\uB97C \uB2E4\uC2DC \uB193\uC73C\uC138\uC694.`,
      uninstallDesktopTip: name => `\uC774 \uC571\uC5D0\uC11C ${name} \uC81C\uAC70`,
      uninstallFailed: name => `${name}\uC744(\uB97C) \uC81C\uAC70\uD558\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4.`,
      uninstallTip: (name, profile) => `${profile}\uC5D0\uC11C ${name} \uC81C\uAC70`,
      uninstalled: name =>
        `${name}\uC774(\uAC00) \uC81C\uAC70\uB418\uC5C8\uC2B5\uB2C8\uB2E4. \uC5B8\uB85C\uB4DC\uD558\uB824\uBA74 \uAC8C\uC774\uD2B8\uC6E8\uC774\uB97C \uC7AC\uC2DC\uC791\uD558\uC138\uC694.`,
      uninstalledDesktop: name => `${name}\uC774(\uAC00) \uC81C\uAC70\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
      updateConsentBody: (name, sha) =>
        `${name} (${sha})\uC758 \uC0C8 \uCE74\uD0C8\uB85C\uADF8 \uACE0\uC815 \uBC84\uC804\uC5D0\uB294 \uC124\uCE58\uB41C \uBC84\uC804\uC5D0 \uC5C6\uB294 \uD45C\uBA74\uC774 \uCD94\uAC00\uB429\uB2C8\uB2E4. \uC2E0\uB8B0\uD558\uB294 \uACBD\uC6B0\uC5D0\uB9CC \uC801\uC6A9\uD558\uC138\uC694.`,
      updateConsentTitle: name => `${name}\uC5D0\uC11C \uCD94\uAC00 \uAD8C\uD55C\uC744 \uC694\uCCAD\uD569\uB2C8\uB2E4.`,
      updateFailed: name =>
        `${name}\uC744(\uB97C) \uC5C5\uB370\uC774\uD2B8\uD558\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4.`,
      updateToPin: sha => `${sha}(\uC73C)\uB85C \uC5C5\uB370\uC774\uD2B8`,
      updated: name =>
        `${name}\uC774(\uAC00) \uD604\uC7AC \uCE74\uD0C8\uB85C\uADF8 \uACE0\uC815 \uBC84\uC804\uC73C\uB85C \uC5C5\uB370\uC774\uD2B8\uB418\uC5C8\uC2B5\uB2C8\uB2E4. \uC801\uC6A9\uD558\uB824\uBA74 \uAC8C\uC774\uD2B8\uC6E8\uC774\uB97C \uC7AC\uC2DC\uC791\uD558\uC138\uC694.`
    },
    toggleToolset: (label, enabled) => `${label} \uD234\uC14B\uC744 ${enabled ? '\uCF1C\uAE30' : '\uB044\uAE30'}`,
    toolsetsEnabled: (enabled, total) => `\uD234\uC14B ${enabled}/${total}\uAC1C \uD65C\uC131\uD654\uB428`,
    usageCount: count => `${count}\xD7 \uC0AC\uC6A9\uB428`
  },
  starmap: {
    importSuccess: nodes =>
      `\uB178\uB4DC ${nodes}\uAC1C\uAC00 \uD3EC\uD568\uB41C \uB9F5\uC744 \uB85C\uB4DC\uD588\uC2B5\uB2C8\uB2E4.`,
    subtitle: (nodes, clusters) => `\uCE74\uD14C\uACE0\uB9AC ${clusters}\uAC1C\uC758 \uC2A4\uD0AC ${nodes}\uAC1C`
  },
  statusStack: {
    background: count => `\uBC31\uADF8\uB77C\uC6B4\uB4DC ${count}\uAC1C`,
    coding: {
      ahead: count => `${count}\uAC1C \uC55E\uC11C \uC788\uC74C`,
      behind: count => `${count}\uAC1C \uB4A4\uCC98\uC838 \uC788\uC74C`,
      branchOffFrom: base => `${base}\uC5D0\uC11C \uC0C8 \uBE0C\uB79C\uCE58 \uC0DD\uC131`,
      changed: count => `\uBCC0\uACBD\uB428 ${count}\uAC1C`,
      commitPlaceholder: shortcut => `\uBA54\uC2DC\uC9C0 (${shortcut}\uB97C \uB20C\uB7EC \uCEE4\uBC0B)`,
      switchFailed: branch => `${branch}(\uC73C)\uB85C \uC804\uD658\uD558\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4.`,
      switchTo: branch => `${branch}(\uC73C)\uB85C \uC804\uD658`
    },
    control: {
      actionFailed: msg => `\uC791\uC5C5 \uC2E4\uD328: ${msg}`,
      controlUnavailable: msg => `\uC138\uC158 \uC81C\uC5B4\uB97C \uC0AC\uC6A9\uD560 \uC218 \uC5C6\uC74C: ${msg}`,
      copyCriterion: index => `\uAE30\uC900 ${index} \uBCF5\uC0AC`,
      criteriaHeader: count => `\uAE30\uC900 \xB7 ${count}`,
      gateAttempts: (attempts, max) => `\uC2DC\uB3C4 ${attempts}/${max}\uD68C`,
      gateLastExit: code => (code === null ? '\uB300\uAE30 \uC911' : `\uC885\uB8CC \uCF54\uB4DC: ${code}`),
      gateTimeout: seconds => `\uD0C0\uC784\uC544\uC6C3 ${seconds}\uCD08`,
      goalActiveTurns: (turn, maxTurns) => `\uD134 ${turn}/${maxTurns}`,
      goalDoneTurns: turns => `\uD134 ${turns}\uAC1C`,
      goalTurn: turn => `\uD134 ${turn}`,
      heartbeatEveryHours: hours => `${hours}\uC2DC\uAC04\uB9C8\uB2E4`,
      heartbeatEveryMinutes: minutes => `${minutes}\uBD84\uB9C8\uB2E4`,
      heartbeatEverySeconds: seconds => `${seconds}\uCD08\uB9C8\uB2E4`,
      heartbeatFiredCount: count => `${count}\uD68C \uC2E4\uD589\uB428`,
      heartbeatNext: time => `\uB2E4\uC74C \uC2E4\uD589: ${time}`,
      loopEveryHours: hours => `${hours}\uC2DC\uAC04\uB9C8\uB2E4`,
      loopEveryMinutes: minutes => `${minutes}\uBD84\uB9C8\uB2E4`,
      loopEverySeconds: seconds => `${seconds}\uCD08\uB9C8\uB2E4`,
      loopNext: time => `\uB2E4\uC74C \uC2E4\uD589: ${time}`,
      loopRunCount: (current, total) => `\uC2E4\uD589 ${current}/${total}`,
      loopRuns: runs => `\uC2E4\uD589 ${runs}\uD68C`,
      removeCriterion: index => `\uAE30\uC900 ${index} \uC81C\uAC70`,
      removeCriterionConfirmBody: index =>
        `\uAE30\uC900 ${index}\uC744(\uB97C) \uC815\uB9D0 \uC81C\uAC70\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
      removeCriterionConfirmTitle: index =>
        `\uAE30\uC900 ${index}\uC744(\uB97C) \uC81C\uAC70\uD558\uC2DC\uACA0\uC2B5\uB2C8\uAE4C?`,
      waitPid: pid => `\uD504\uB85C\uC138\uC2A4 ${pid} \uB300\uAE30 \uC911`,
      waitSession: target => `\uC138\uC158 ${target} \uB300\uAE30 \uC911`,
      waitUntil: target => `${target}\uAE4C\uC9C0 \uB300\uAE30`
    },
    exit: code => `\uC885\uB8CC ${code}`,
    previousTodos: (done, total) => `\uC774\uC804 \uC791\uC5C5 ${done}/${total}`,
    subagents: count => `\uD558\uC704 \uC5D0\uC774\uC804\uD2B8 ${count}\uAC1C`,
    todos: (done, total) => `\uC791\uC5C5 ${done}/${total}`
  },
  updates: {
    applyStatus: {
      owed: steps => `업데이트되었지만 아직 완료되지 않은 항목: ${steps}. \`hermes update\`를 다시 실행해 완료하세요.`
    }
  },
  webhooks: {
    createFailed: detail => `\uC0DD\uC131 \uC2E4\uD328: ${detail}`,
    deleteFailed: name => `"${name}" \uC0AD\uC81C \uC2E4\uD328`,
    disabled: name => `\uBE44\uD65C\uC131\uD654\uB428: "${name}"`,
    enabled: name => `\uD65C\uC131\uD654\uB428: "${name}"`,
    restartFailed: detail => `\uAC8C\uC774\uD2B8\uC6E8\uC774 \uC7AC\uC2DC\uC791 \uC2E4\uD328${detail}`,
    subscriptions: count => `\uAD6C\uB3C5 (${count})`,
    toggleFailed: (name, enabled) =>
      `"${name}"\uC744(\uB97C) ${enabled ? '\uCF1C\uC9C0' : '\uB044\uC9C0'} \uBABB\uD588\uC2B5\uB2C8\uB2E4.`
  },
  zones: {
    customZoneName: count => `\uC0AC\uC6A9\uC790 \uC9C0\uC815 ${count}\uAC1C \uAD6C\uC5ED`,
    deletePreset: name => `${name} \uC0AD\uC81C`,
    hideStripTab: title => `${title} \uC228\uAE30\uAE30`,
    layoutNamePlaceholder: fallback => `\uB808\uC774\uC544\uC6C3 \uC774\uB984 (${fallback})`,
    mergeZones: count => `\uAD6C\uC5ED ${count}\uAC1C \uBCD1\uD569`,
    missingPane: paneId => `\uB204\uB77D\uB41C \uCC3D: ${paneId}`,
    pluginDisabled: pluginId =>
      `\uD50C\uB7EC\uADF8\uC778 "${pluginId}"\uC774(\uAC00) \uBE44\uD65C\uC131\uD654\uB418\uC5C8\uC2B5\uB2C8\uB2E4.`,
    showStripTab: title => `${title} \uD45C\uC2DC`,
    tabCount: count => `\uD0ED ${count}\uAC1C`,
    toggleStripTab: title => `${title} \uD0ED \uC804\uD658`,
    zoneCount: count => `\uAD6C\uC5ED ${count}\uAC1C`,
    zoneMenuLabel: title => `${title} \uAD6C\uC5ED \uC635\uC158`,
    zoneTag: index => `\uAD6C\uC5ED ${index}`
  }
} satisfies TranslationOverrides
