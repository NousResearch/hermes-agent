# Kotlin Test Template (KotlinTest + Mockk)

```kotlin
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class ExampleClassTest {

    @Test
    fun `method should return expected value`() {
        // Given
        val mockDependency = mockk<Dependency>()
        every { mockDependency.getValue() } returns "expected"
        val sut = ExampleClass(mockDependency)

        // When
        val result = sut.method()

        // Then
        assertEquals("expected", result)
        verify { mockDependency.getValue() }
    }
}
```

## Patterns

- Use KotlinTest or JUnit 5
- Use Mockk for mocks
- Given-When-Then pattern
- Verify all mock calls
- Use backticks for descriptive test names
