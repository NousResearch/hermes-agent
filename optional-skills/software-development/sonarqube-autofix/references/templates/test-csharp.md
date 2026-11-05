# C# Test Template (xUnit + Moq)

```csharp
using Xunit;
using Moq;

namespace Example.Tests
{
    public class ExampleClassTests
    {
        private readonly Mock<IDependency> _mockDependency;
        private readonly ExampleClass _sut;

        public ExampleClassTests()
        {
            _mockDependency = new Mock<IDependency>();
            _sut = new ExampleClass(_mockDependency.Object);
        }

        [Fact]
        public void Method_ShouldReturnExpectedValue()
        {
            // Given
            _mockDependency.Setup(x => x.GetValue()).Returns("expected");

            // When
            var result = _sut.Method();

            // Then
            Assert.Equal("expected", result);
            _mockDependency.Verify(x => x.GetValue(), Times.Once);
        }
    }
}
```

## Patterns

- Use xUnit
- Use Moq for mocks
- Given-When-Then pattern
- Verify all mock calls
- Use constructor for setup
