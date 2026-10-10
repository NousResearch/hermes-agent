# Java Test Coverage

Target: **85% line and branch** coverage. Two build tools: **Maven** and **Gradle**. Both use **JaCoCo** for instrumentation and reporting.

## Maven (pom.xml)

### Run with JaCoCo

```bash
# prepare agent + run tests + generate report
mvn -Dmaven.repo.local=./.m2/repository clean test jacoco:report
```

Report: `target/site/jacoco/index.html`. XML: `target/site/jacoco/jacoco.xml`.

### Bind JaCoCo in the build (pom.xml)

```xml
<plugin>
  <groupId>org.jacoco</groupId>
  <artifactId>jacoco-maven-plugin</artifactId>
  <version>0.8.12</version>
  <executions>
    <execution>
      <goals>
        <goal>prepare-agent</goal>
        <goal>report</goal>
      </goals>
    </execution>
    <execution>
      <id>check</id>
      <goals><goal>check</goal></goals>
      <configuration>
        <rules>
          <rule>
            <limits>
              <limit>
                <counter>LINE</counter>
                <value>COVEREDRATIO</value>
                <minimum>0.85</minimum>
              </limit>
              <limit>
                <counter>BRANCH</counter>
                <value>COVEREDRATIO</value>
                <minimum>0.85</minimum>
              </limit>
            </limits>
          </rule>
        </rules>
      </configuration>
    </execution>
  </executions>
</plugin>
```

Then `mvn verify` fails the build if coverage drops below 85%.

## Gradle (build.gradle / build.gradle.kts)

### Run with JaCoCo

```bash
./gradlew test jacocoTestReport
```

Report: `build/reports/jacoco/test/html/index.html`.

### Apply the plugin (build.gradle)

```groovy
plugins {
  id 'jacoco'
}
jacoco { toolVersion = "0.8.12" }
jacocoTestReport {
  reports {
    xml.required = true
    html.required = true
  }
}
test.finalizedBy jacocoTestReport
```

### Enforce a threshold (build.gradle)

```groovy
jacocoTestCoverageVerification {
  violationRules {
    rule {
      limit {
        counter = 'LINE'
        minimum = 0.85
      }
      limit {
        counter = 'BRANCH'
        minimum = 0.85
      }
    }
  }
}
check.finalizedBy jacocoTestCoverageVerification
```

## Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| `jacoco:report` with no `prepare-agent` | Empty/0% report | Run `jacoco:prepare-agent` before `test`, or bind in pom |
| Report missing after `mvn test` | JaCoCo not bound | Add `jacoco-maven-plugin` execution `report` |
| Gradle report not generated | `jacocoTestReport` not invoked | Add `test.finalizedBy jacocoTestReport` |
