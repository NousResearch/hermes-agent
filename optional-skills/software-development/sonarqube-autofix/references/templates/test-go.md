# Go Test Template

```go
package example_test

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
)

type MockDependency struct {
    mock.Mock
}

func (m *MockDependency) GetValue() string {
    args := m.Called()
    return args.String(0)
}

func TestExampleMethod(t *testing.T) {
    // Given
    mockDep := new(MockDependency)
    mockDep.On("GetValue").Return("expected")
    sut := NewExample(mockDep)

    // When
    result := sut.Method()

    // Then
    assert.Equal(t, "expected", result)
    mockDep.AssertExpectations(t)
}
```

## Patterns

- Use testing
- Use testify/mock for mocks
- Use testify/assert for assertions
- Given-When-Then pattern
- Verify all mock calls
