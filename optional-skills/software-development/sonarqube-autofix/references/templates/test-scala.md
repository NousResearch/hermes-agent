# Scala Test Template (ScalaTest + Mockito)

```scala
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.mockito.MockitoSugar
import org.mockito.Mockito._

class ExampleClassSpec extends AnyFlatSpec with Matchers with MockitoSugar {

  "ExampleClass" should "return expected value" in {
    // Given
    val mockDependency = mock[Dependency]
    when(mockDependency.getValue()).thenReturn("expected")
    val sut = new ExampleClass(mockDependency)

    // When
    val result = sut.method()

    // Then
    result shouldBe "expected"
    verify(mockDependency).getValue()
  }
}
```

## Patterns

- Use ScalaTest
- Use Mockito for mocks
- Given-When-Then pattern
- Verify all mock calls
- Use shouldMatchers for assertions
